"""Fine sweep between point_scale=0.65 and 0.95 using subpixel splats.

Run directly in an IDE.  Unlike the default integer-radius renderer, the
fractional boundary keeps intermediate point scales visually distinguishable.
"""

from plot_pcv_edge_edl_point_size import run


POINT_SCALES = (0.68, 0.74, 0.80, 0.86, 0.92)


if __name__ == "__main__":
    run(
        point_scales=POINT_SCALES,
        fractional_splat=True,
        output_suffix="_fine",
    )
