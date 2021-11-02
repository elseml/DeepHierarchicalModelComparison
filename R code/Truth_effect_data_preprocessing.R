library(tidyverse)

### Download and preprocess truth effect data 
# code from https://github.com/PerceptionAndCognitionLab/hc-truth/blob/public/papers/sub/p.Rmd

dat1 <- read_csv(url("https://osf.io/cvpmr/download")) %>% 
  dplyr::filter(runningtrial == "TruthJudgment") %>% 
  dplyr::select(subject, statement, status, repeated, truthrating) %>% 
  dplyr::mutate(item = ifelse(status, 0.5, -0.5),
                sub = as.integer(as.factor(subject)),
                cond = ifelse(repeated == "No", -0.5, 0.5),
                Y = truthrating) %>% 
  dplyr::select(sub, item, cond, Y) %>% 
  dplyr::mutate(Y = (Y - 3)/2)

dat3 <- read_csv(url("https://osf.io/4znvx/download")) %>% 
  dplyr::filter(phase == 2 & filter == "selected" & group == 2) %>% 
  dplyr::mutate(item = ifelse(status, 0.5, -0.5),
                sub = as.integer(as.factor(subject)),
                cond = ifelse(repetition == "No", -0.5, 0.5),
                Y = trating) %>% 
  dplyr::select(sub, item, cond, Y) %>% 
  dplyr::mutate(Y = (Y - 3.5)/2.5) 

dat5 <- read_csv(url("https://osf.io/qsrw5/download")) %>% 
  dplyr::filter(phase == 3) %>% 
  dplyr::mutate(item = ifelse(status, 0.5, -0.5),
                sub = as.integer(as.factor(subject)),
                cond = ifelse(repeated_phase3 == "No", -0.5, 0.5),
                Y = trating) %>% 
  dplyr::select(sub, item, cond, Y) %>% 
  dplyr::mutate(Y = (Y - 3.5)/2.5)


### Set working directory to be the folder above the one in which the script resides
setwd(dirname(dirname(rstudioapi::getSourceEditorContext()$path)))

### Save data
write.table(dat1, file="data/Truth_effect_example/te_data1")

write.table(dat3, file="data/Truth_effect_example/te_data3")

write.table(dat5, file="data/Truth_effect_example/te_data5")
