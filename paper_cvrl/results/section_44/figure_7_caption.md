**Fig. 7. Conditional gains and pairing validity of return-structure representations.**
**(a)** IoU gain of the full CVRL representation over its visual-only branch
across regional visibility, ground, vegetation, and building conditions.
Ground and vegetation results are class-IoU gains computed within spatial
evaluation windows containing all target and predicted classes; building
facade and roof use class IoU over the fixed test set. Region-level results use
mIoU. Points denote estimates and horizontal bars denote 95% confidence
intervals obtained by source-block bootstrap. **(b)** Frozen linear-probe and
cosine-kNN mIoU for correctly paired `zV || zR`, `zV || Random-72D`, and
source-block-wise `zV || Shuffled-zR`. Random and shuffled controls report the
mean and standard deviation over five seeds; the correct pair is fixed.
