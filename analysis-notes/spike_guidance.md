# Information for Spike

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

