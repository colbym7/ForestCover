library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(corrplot)
library(ggthemes)
library(lightgbm)
library(bonsai)
library(finetune)
library(ranger)
library(stacks)


  # Read in Data
train <- vroom('C:\\Users\\cjmsp\\Desktop\\Stat348\\ForestCover\\train.csv') %>%
  mutate(Cover_Type = factor(Cover_Type))
test <- vroom('C:\\Users\\cjmsp\\Desktop\\Stat348\\ForestCover\\test.csv')

  # ELU Feature Engineering
soil_cols <- paste0("Soil_Type", 1:40)
soil_index_train <- max.col(train[, soil_cols])
soil_index_test  <- max.col(test[, soil_cols])


ELU_CODE <- c(
  "1"=2702,"2"=2703,"3"=2704,"4"=2705,"5"=2706,"6"=2717,"7"=3501,
  "8"=3502,"9"=4201,"10"=4703,"11"=4704,"12"=4744,"13"=4758,"14"=5101,
  "15"=5151,"16"=6101,"17"=6102,"18"=6731,"19"=7101,"20"=7102,"21"=7103,
  "22"=7201,"23"=7202,"24"=7700,"25"=7701,"26"=7702,"27"=7709,"28"=7710,
  "29"=7745,"30"=7746,"31"=7755,"32"=7756,"33"=7757,"34"=7790,"35"=8703,
  "36"=8707,"37"=8708,"38"=8771,"39"=8772,"40"=8776
)
train$elu <- ELU_CODE[as.character(soil_index_train)]
train$elu <- as.integer(train$elu)
train$elu_zone    <- train$elu %/% 1000                # A
train$elu_subzone <- (train$elu %/% 100) %% 10         # B
train$elu_class   <- (train$elu %/% 10) %% 10          # C
train$elu_variant <- train$elu %% 10   


test$elu <- ELU_CODE[as.character(max.col(test[, soil_cols]))]
test$elu <- as.integer(test$elu)
test$elu_zone    <- test$elu %/% 1000
test$elu_subzone <- (test$elu %/% 100) %% 10
test$elu_class   <- (test$elu %/% 10) %% 10
test$elu_variant <- test$elu %% 10

train$Soil_Type <- factor(soil_index_train, levels = 1:40)
test$Soil_Type  <- factor(soil_index_test,  levels = 1:40)
train$Soil_Type <- as.numeric(train$Soil_Type)
test$Soil_Type <- as.numeric(test$Soil_Type)

train <- train %>% select(-all_of(soil_cols))
test  <- test %>% select(-all_of(soil_cols))



  # EDA
summary(train$Cover_Type)
glimpse(train)
skimr::skim(train)

ggplot(data = train, aes(x = Elevation, group = factor(Cover_Type), colour = factor(Cover_Type))) +
  geom_density() +
  scale_color_colorblind()

ggplot(data = train, aes(x = Aspect, group = factor(Cover_Type), colour = factor(Cover_Type))) +
  geom_density() +
  scale_color_colorblind()

ggplot(data = train, aes(x = Slope, group = factor(Cover_Type), colour = factor(Cover_Type))) +
  geom_density() +
  scale_color_colorblind()

corrplot(cor(train[c(1:10)]))
plot(density(train$Aspect))
summary(train$Aspect)
summary(train$Hillshade_Noon)


  # Further Feature Engineering
one_eighty <- function(x) {
  ifelse(x + 180 > 360, x - 180, x + 180)
}

fe <- function(df) {
  df <- df %>%
    mutate(
      HdrElev = Horizontal_Distance_To_Roadways * Elevation,
      VdhElev = Vertical_Distance_To_Hydrology * Elevation,
      Aspect2 = one_eighty(Aspect),
      
      Aspect = ifelse(Aspect < 0, Aspect + 360, Aspect),
      Aspect = ifelse(Aspect > 359, Aspect - 360, Aspect),
      
      # Limit Hillshade values to [0, 255]
      Hillshade_9am  = pmin(pmax(Hillshade_9am, 0), 255),
      Hillshade_Noon = pmin(pmax(Hillshade_Noon, 0), 255),
      Hillshade_3pm  = pmin(pmax(Hillshade_3pm, 0), 255),
      
      sin_aspect = sin(Aspect),
      Highwater = as.integer(Vertical_Distance_To_Hydrology < 0),
      EVDtH = Elevation - Vertical_Distance_To_Hydrology,
      EHDtH = Elevation - Horizontal_Distance_To_Hydrology * 0.2,
      TotalDistancetoHydrolody =
        sqrt(Horizontal_Distance_To_Hydrology^2 +
               Vertical_Distance_To_Hydrology^2),
      SumDistancetoHydrolody =
        Horizontal_Distance_To_Hydrology +
        Vertical_Distance_To_Hydrology,
      
      Hydro_Fire_1 =
        Horizontal_Distance_To_Hydrology +
        Horizontal_Distance_To_Fire_Points,
      Hydro_Fire_2 =
        abs(Horizontal_Distance_To_Hydrology -
              Horizontal_Distance_To_Fire_Points),
      
      Hydro_Road_1 =
        abs(Horizontal_Distance_To_Hydrology +
              Horizontal_Distance_To_Roadways),
      Hydro_Road_2 =
        abs(Horizontal_Distance_To_Hydrology -
              Horizontal_Distance_To_Roadways),
      
      Fire_Road_1 =
        abs(Horizontal_Distance_To_Fire_Points +
              Horizontal_Distance_To_Roadways),
      Fire_Road_2 =
        abs(Horizontal_Distance_To_Fire_Points -
              Horizontal_Distance_To_Roadways),
      
      Hillshade_3pm_is_zero = as.integer(Hillshade_3pm == 0),
      slope_elev_ratio = Slope/Elevation
    )
}
train <- fe(train)
test <- fe(test)


    ### Stacked Model (RF, XGB, Multi-Logistic Reg)
my_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  step_zv() %>%
  step_nzv() %>%
  step_normalize(all_numeric_predictors())

rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  mtry = tune(),
  sample_size = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

glm_spec <- multinom_reg(
  penalty = tune(),   # lambda
  mixture = tune()    # 0 = ridge, 1 = lasso, in between = elastic net
) %>%
  set_engine("glmnet")

rf_wf  <- workflow() %>% add_model(rf_spec)  %>% add_recipe(my_recipe)
xgb_wf <- workflow() %>% add_model(xgb_spec) %>% add_recipe(my_recipe)
glm_wf <- workflow() %>% add_model(glm_spec) %>% add_recipe(my_recipe)


folds <- vfold_cv(train, v = 5, repeats = 1)
ctrl <- control_stack_resamples()
rf_res  <- tune_grid(rf_wf,  resamples = folds, grid = 5, control = ctrl)
xgb_res <- tune_grid(xgb_wf, resamples = folds, grid = 5, control = ctrl)
glm_res <- tune_grid(glm_wf, resamples = folds, grid = 5, control = ctrl)

model_stack <- stacks() %>%
  add_candidates(rf_res) %>%
  add_candidates(xgb_res) %>%
  add_candidates(glm_res)
stack_blended <- blend_predictions(model_stack)
stack_fit <- fit_members(stack_blended)
stack_preds <- predict(stack_fit, new_data = test, type = "class")

kaggle_submission1 <- data.frame(
  id = test$Id,
  Cover_Type = stack_preds$.pred_class
)
vroom_write(x=kaggle_submission1, 
            file="C://Users//cjmsp//Desktop//Stat348//ForestCover//Preds//stack_preds.csv", 
            delim=",")


    ### Light Gradient Boosted Model
lgbm_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune(),
  loss_reduction = tune()
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

lgbm_wf <- workflow() %>%
  add_model(lgbm_spec) %>%
  add_recipe(my_recipe)

folds <- vfold_cv(train, v=5)
lgbm_grid <- grid_space_filling(
  trees(),
  tree_depth(),
  learn_rate(range = c(-4, -1)),   # log10 scale
  min_n(),
  loss_reduction(),
  size = 20
)
tuned_results <- lgbm_wf %>%
  tune_race_anova(
    resamples = folds,
    grid = lgbm_grid,
    metrics = metric_set(accuracy)
  )
best_tune <- tuned_results %>%
  select_best(metric = 'accuracy')

final_lgbm_wf <- lgbm_wf %>%
  finalize_workflow(best_tune)

final_lgbm_fit <- final_lgbm_wf %>%
  fit(data = train)
lgbm_preds <- predict(final_lgbm_fit, test) %>%
  bind_cols(test)

kaggle_submission2 <- data.frame(
  id = test$Id,
  Cover_Type = lgbm_preds$.pred_class
)
vroom_write(x=kaggle_submission2, 
            file="C://Users//cjmsp//Desktop//Stat348//ForestCover//Preds//lgbm_preds.csv", 
            delim=",")


    ### Random Forest ###
soil_vars <- paste0('Soil_Type', 1:40)
train_target <- train %>%
  mutate(soil_type = soil_vars[apply(select(., all_of(soil_vars)), 1, which.max)])
test_target <- test %>%
  mutate(soil_type = soil_vars[apply(select(., all_of(soil_vars)), 1, which.max)])

recipe1 <- recipe(Cover_Type ~ ., data = train_target) %>%
  step_mutate(total_distance_hydro = sqrt((Horizontal_Distance_To_Hydrology^2)+ (Vertical_Distance_To_Hydrology^2))) %>%
  step_rm(starts_with("Soil_Type"))

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(recipe1) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range=c(1,16)),
                            min_n(),
                            levels = 4)

folds <- vfold_cv(train_target, v = 5, repeats = 1)
CV_results <- rf_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(accuracy))
bestTune <- CV_results %>%
  select_best(metric = 'accuracy')

final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_target)

rf_preds <- final_wf %>%
  predict(new_data=test_target, type='class')

kaggle_submission3 <- data.frame(
  id = test$Id,
  Cover_Type = rf_preds$.pred_class
)
vroom_write(x=kaggle_submission3, 
            file="C://Users//cjmsp//Desktop//Stat348//ForestCover//Preds//rftarget_preds.csv", 
            delim=",")


    ### Data Robot ###
my_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  step_zv() %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(my_recipe)
baked_data <- bake(prepped_recipe, new_data = NULL)
write.csv(baked_data, file = 'C:\\Users\\cjmsp\\Desktop\\Stat348\\ForestCover\\datarobot.csv')
write.csv(test, file = 'C:\\Users\\cjmsp\\Desktop\\Stat348\\ForestCover\\datarobot_test.csv')

robot_test <- read.csv('C:\\Users\\cjmsp\\Desktop\\Stat348\\ForestCover\\datarobotoutput.csv')
#robot_test  <- bake(prepped_recipe, new_data = robot_test)
write.csv(robot_test, file = 'C:\\Users\\cjmsp\\Desktop\\Stat348\\ForestCover\\datarobot_test.csv')

kaggle_submission4 <- data.frame(
  id = robot_test$Id,
  Cover_Type = robot_test$Cover_Type_PREDICTION
)
vroom_write(x=kaggle_submission4, 
            file="C://Users//cjmsp//Desktop//Stat348//ForestCover//Preds//robot_preds.csv", 
            delim=",")


recipe4 <- recipe(cuisine ~ ., data = train) %>%
  step_mutate(
    n_ingredients = map_int(ingredients, length),
    is_dairy = map_lgl(
      ingredients,
      ~ any(str_detect(.x, regex("milk|cheese|cream|butter", ignore_case = TRUE)))
    ),
    has_peppers = map_lgl(
      ingredients,
      ~ any(str_detect(.x, regex("pepper|chili|chilli|capsicum", ignore_case = TRUE)))
    )
  ) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens=1000) %>%
  step_tfidf(ingredients)
prepped_recipe <- prep(recipe4)
baked_data <- bake(prepped_recipe, new_data = NULL)
baked_2 <- bake(prepped_recipe, new_data = test)
write.csv(baked_data, file = 'C:\\Users\\cjmsp\\Desktop\\Stat348\\WhatsCooking\\datarobot.csv')
write.csv(baked_2, file = 'C:\\Users\\cjmsp\\Desktop\\Stat348\\WhatsCooking\\datarobottest.csv')

robot_preddf <- read.csv('C:\\Users\\cjmsp\\Desktop\\Stat348\\WhatsCooking\\robotresult.csv')
kaggle_submission6 <- data.frame(
  id = test$id,
  cuisine = robot_preddf$cuisine_PREDICTION
)
vroom_write(x=kaggle_submission6, 
            file="C://Users//cjmsp//Desktop//Stat348//WhatsCooking//Preds//robotnn1_preds.csv", 
            delim=",")
