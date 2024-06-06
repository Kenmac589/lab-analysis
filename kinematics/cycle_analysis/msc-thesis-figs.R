library(readr)
library(ggpubr)
library(ggplot2)

# Read in data
moddf <- read_csv("./mos_limbs_combined_all.csv", col_names = TRUE)

mos_violin <- ggviolin(moddf,
  x = "Condition", y = "MoS", color = "Perturbation State",
  palette = c("#00AFBB", "#E7B800", "#FC4E07"),
  add = "jitter", shape = "Condition"
)

custom_p_format <- function(p) {
  rstatix::p_format(p, accuracy = 0.0001, digits = 3, leading.zero = FALSE)
}
anova_mos <- mos_violin + stat_anova_test(aes(group = Condition),
  label = "Anova, italic(p) = {custom_p_format(p)}{p.signif}"
)


ggexport(anova_mos, filename = "./anova_mos.png", width = 1920, height = 1080, res = 150)
