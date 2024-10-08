library(readr)
library(ggpubr)
library(ggprism)
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstatix)

# Read in data
moddf <- read_csv("./mos_limbs_combined_all.csv", col_names = TRUE)
print(moddf)
colnames(moddf) <- gsub(" (cm)", "", colnames(moddf), fixed = TRUE)
print(moddf)

stat.test <- aov(MoS ~ Condition, data = moddf) |> tukey_hsd()

stat.test

# mos_violin <- ggviolin(moddf,
#   x = "Condition", y = "MoS", color = "Perturbation State",
#   palette = c("#00AFBB", "#E7B800", "#FC4E07"),
#   add = "jitter", shape = "Condition"
# ) + theme_prism()
#
# custom_p_format <- function(p) {
#   rstatix::p_format(p, accuracy = 0.0001, digits = 3, leading.zero = FALSE)
# }

anova_box <- ggboxplot(moddf, x = "Condition", y = "MoS") +
  stat_pvalue_manual(
    stat.test,
    label = "p.adj",
    y.position = c(2.5, 2.6, 2.7, 2.8, 2.9, 3.0)
  )
anova_box

anova_mos <- mos_violin + stat_pvalue_manual(stat.test,
  label = "p.adj",
  y.position = c(2.5, 2.6, 2.6, 2.7)
)

anova_mos

ggexport(anova_mos,
  filename = "./r_figures/anova_mos.svg",
  width = 15,
  # height = 1080,
  res = 300
)
