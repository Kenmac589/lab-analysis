# Information for Spike

[saved_spike_pages](saved_spike_pages.md)

- Aim for between 15-20 step cycles

1. DC Remove -> remove zero
2. Rectify
3. Smoothen
    - Going with a smoothening factor of 0.01

When creating channels like sw onset
- Create new memory buffer
- Take ToeX and add the process of slope
- Import toex channel with every time it rises through

- Also for cleaning up random blips on DLC, you the channel process `median filter`

## For Post Factorization Primitive Cleanup

- Import factorized primitives in to spike at file sampling rate of 1000 Hz
- Make sure comma is set as the delimiter
- Smoothen to 0.01
- Consistently trim first and last 0.1 samples by setting cursors on either end
- Make sure you select the channels in order starting from the first ending with the last
- Export settings:
    - Output sample rate of 1000 Hz to maintain same amount of samples
    - Time range based on cursors
    - Use Spline waveform interpolation to ensure smooth lines
    - Use Comma Separator

**After export from spike make sure to rearrange channel order appropriately based on the channel titles if not done already**
