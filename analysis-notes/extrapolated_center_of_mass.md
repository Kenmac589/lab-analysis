# Relating to xCoM

## xCoM

- Based on Turgay's import script, you should have at least the following channels visible:
    - First three . . . duh
    - 37 CoMy
    - 35 FRy
    - 33 FLy
    - 30 HRy
    - 28 HLy
    - Whatever your right and left double support phase channels are.
- In the file I worked on with Turgay the equation has
    - Ch(6006) + (Ch(37) / Sqrt(9.81/0.02777))
    - Ch(6006) : Also a random channel Turgay made which is just the CoMy
    - Ch(37) : Is a smoothened and sloped version of CoMy
        - The smoothening factor applied here 0.02 (damn).
    - Revised = `Ch(37) + (Ch(6001)/Sqrt(9.81/0.0214))`
        - `Ch(37)` is just CoMy
        - `Ch(6002)` is a smoothened and sloped version of CoMy
        - `0.0214` is the hip_height
## Center of Pressure

- For centre of pressure
    - For the **right** side it will be 
        - `((Ch(30) + Ch(35)) / 2) * Ch(60)` for Egr3/WT
        - `((Ch(36) + Ch(40)) / 2) * Ch(51)` for DTR 2-3
        - `((Ch(36) + Ch(40)) / 2) * Ch(49)` for DTR 5 pre
    - For the **left** side it would be
        - `((Ch(33) + Ch(28)) / 2) * Ch(59)` for Egr3/WT
        - `((Ch(38) + Ch(34)) / 2) * Ch(52)` for Egr3/WT
        - `((Ch(38) + Ch(34)) / 2) * Ch(50)` for DTR 5 pre
    - These are based on the y coordinates for the respective limbs multiplied by the COP level marker 
- For double support
    - Forelimb sw of to hindlimb swon
    - level type of channel
    - import combined event event channel buffer into level
