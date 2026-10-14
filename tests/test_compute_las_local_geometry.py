import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from utils.compute_las_local_geometry import (
    DESCRIPTOR_NAMES, NORMAL_NAMES, VALID_NAME,
    compute_las_local_geometry, compute_local_geometry,
    compute_local_geometry_gpu, process_file,
)


class TestOfflineLasGeometry(unittest.TestCase):
    def test_gpu_matches_cpu_with_large_map_coordinates(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(8)
        xyz = rng.normal(size=(3000, 3)) * 2 + [2_600_000, 1_200_000, 500]
        cpu = compute_local_geometry(xyz, k=16)
        gpu = compute_local_geometry_gpu(xyz, k=16, chunk_size=1024)
        np.testing.assert_array_equal(cpu[2], gpu[2])
        self.assertLess(float(np.mean(np.abs(cpu[1] - gpu[1]))), 1e-4)
        self.assertGreater(float(np.mean(np.abs(np.sum(cpu[0] * gpu[0], axis=1)))), .999)

    def test_plane_line_curved_and_scale(self):
        grid = np.linspace(-2, 2, 23)
        x, y = np.meshgrid(grid, grid)
        plane = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        normal, shape, valid = compute_local_geometry(
            plane, k=32, reference_grid_size=0
        )
        self.assertTrue(valid.all())
        self.assertGreater(float(np.median(np.abs(normal[:, 2]))), .99)
        self.assertLess(float(np.quantile(shape[:, 3], .95)), 1e-5)
        self.assertGreater(float(np.median(shape[:, 1])), .2)
        _, scaled, _ = compute_local_geometry(plane * 4, k=32, reference_grid_size=0)
        np.testing.assert_allclose(shape, scaled, atol=1e-5)
        line = np.column_stack((np.linspace(-4, 4, 200), np.zeros(200), np.zeros(200)))
        _, line_shape, _ = compute_local_geometry(line, k=32, reference_grid_size=0)
        self.assertGreater(float(np.median(line_shape[:, 0])), .99)
        self.assertLess(float(np.max(line_shape[:, 1])), 1e-5)
        rng = np.random.default_rng(5)
        sphere = rng.normal(size=(800, 3))
        sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
        _, curved, _ = compute_local_geometry(sphere, k=48, reference_grid_size=0)
        self.assertGreater(float(np.median(curved[:, 3])), 0)
        self.assertTrue(np.all((curved >= 0) & (curved <= 1)))

    def test_las_round_trip_retains_existing_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.las"
            output = Path(folder) / "output.las"
            rng = np.random.default_rng(42)
            xyz = rng.uniform(0, 10, (300, 3))
            header = laspy.LasHeader(point_format=3, version="1.2")
            cloud = laspy.LasData(header)
            cloud.x, cloud.y, cloud.z = xyz.T
            cloud.classification = rng.integers(1, 7, 300, dtype=np.uint8)
            cloud.intensity = rng.integers(1, 1000, 300, dtype=np.uint16)
            cloud.return_number = np.ones(300, dtype=np.uint8)
            cloud.number_of_returns = np.ones(300, dtype=np.uint8)
            cloud.write(source)
            result = process_file(source, output, k=16, reference_grid_size=.25,
                                  chunk_size=64, overwrite=False)
            self.assertEqual(result["status"], "written")
            original, modified = laspy.read(source), laspy.read(output)
            self.assertEqual(len(original.points), len(modified.points))
            for name in ("classification", "intensity", "return_number", "number_of_returns"):
                np.testing.assert_array_equal(np.asarray(original[name]), np.asarray(modified[name]))
            np.testing.assert_array_equal(np.asarray(original.X), np.asarray(modified.X))
            for name in (*NORMAL_NAMES, *DESCRIPTOR_NAMES, VALID_NAME):
                self.assertIn(name, modified.point_format.dimension_names)
                self.assertTrue(np.isfinite(np.asarray(modified[name])).all())
            self.assertTrue(np.all(np.asarray(modified[VALID_NAME]) == 1))
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except ImportError:
                cuda_available = False
            if cuda_available:
                gpu_output = Path(folder) / "output_gpu.las"
                process_file(source, gpu_output, k=16, reference_grid_size=.25,
                             chunk_size=128, overwrite=False, backend="gpu")
                accelerated = laspy.read(gpu_output)
                np.testing.assert_array_equal(np.asarray(original.X), np.asarray(accelerated.X))
                np.testing.assert_array_equal(np.asarray(original.classification),
                                              np.asarray(accelerated.classification))
                cpu_shape = np.column_stack([np.asarray(modified[name]) for name in DESCRIPTOR_NAMES])
                gpu_shape = np.column_stack([np.asarray(accelerated[name]) for name in DESCRIPTOR_NAMES])
                self.assertLess(float(np.mean(np.abs(cpu_shape - gpu_shape))), 1e-4)

    def test_folder_gpu_resume_ignores_partial_files(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        with tempfile.TemporaryDirectory() as folder:
            source_dir = Path(folder) / "source"
            output_dir = Path(folder) / "output"
            source_dir.mkdir()
            cloud = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
            xyz = np.random.default_rng(6).uniform(0, 5, (200, 3))
            cloud.x, cloud.y, cloud.z = xyz.T
            cloud.write(source_dir / "tile.las")
            cloud.write(source_dir / ".interrupted.las")
            options = dict(k=16, reference_grid_size=.25, chunk_size=128,
                           backend="gpu", num_workers=2)
            first = compute_las_local_geometry(source_dir, output_dir, **options)
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0]["status"], "written")
            second = compute_las_local_geometry(source_dir, output_dir, **options)
            self.assertEqual(second[0]["status"], "skipped_existing")


if __name__ == "__main__":
    unittest.main()
