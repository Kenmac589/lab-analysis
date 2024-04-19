# Analysis results

## Two way anova between egr3 and wt

- Needing to check best method for doing this but I can't imagine a more proper method as there are really only two things to be fed in.
    - Mouse Strain
    - Perturbation State

Two way ANOVA done with `statsmodels` 
                           df     sum_sq   mean_sq          F        PR(>F)
C(mouse_type)             1.0   6.150229  6.150229  68.315043  1.059932e-15
C(state)                  2.0   1.452784  0.726392   8.068564  3.520145e-04
C(mouse_type):C(state)    2.0   0.447593  0.223796   2.485869  8.419328e-02
Residual                547.0  49.245012  0.090027        NaN           NaN

Two way Anova done with `pingouin`
               Source         SS     DF        MS          F         p-unc       np2
0          mouse_type   5.228999    1.0  5.228999  58.082282  1.116254e-13  0.095991
1               state   1.452784    2.0  0.726392   8.068564  3.520145e-04  0.028656
2  mouse_type * state   0.447593    2.0  0.223796   2.485869  8.419328e-02  0.009007
3            Residual  49.245012  547.0  0.090027        NaN           NaN       NaN

- Wanted to check the effect of using one packages ANOVA vs the other
