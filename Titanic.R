library(tidyverse)
library(stringr)
library(broom) #for glance function
library(zoo)
library(modeest)
library(pscl) # McFadden R^2 index - used to assess model fit for glm models
library(ROCR) # ROC curve

#import data frames
path <- "C:/Users/Cat/Google Drive/Kitty/Kaggle Competitions/Titanic - Machine Learning/"
  
df_train <- read_csv(file=str_c(path, "train.csv"))
df_test <- read_csv(file=str_c(path, "test.csv"))

#review data frame
df_train %>% glimpse() %>% summary()

df_train$Survived %>% table()

sapply(df_train, function(x) sum(is.na(x)))

sapply(df_train, function(x) length(unique(x)))

Amelia::missmap(df_train, main = "Missing values vs observed")

#clean training data
df_train <- df_train %>% mutate(Survived = as.logical(Survived),
                                PassengerId = as.integer(PassengerId),
                                Pclass = as.factor(Pclass),
                                Sex = if_else(Sex == 'female', 1, 2) %>% as.factor(),
                                SibSp = as.ordered(SibSp),
                                Parch = as.ordered(Parch),
                                TicketNbr = str_extract(Ticket, "[:digit:]*$") %>% as.integer(),
                                Ticket = as.factor(Ticket),
                                Embarked = as.factor(Embarked),
                                Lname = str_extract(Name, "^.*(?=,)") %>% as.factor(),
                                Title = str_extract(Name, "(?<=,\\s)[:alpha:]+\\s?[:alpha:]*(?=.)") %>% as.factor()) %>%
  select(-Name, -Cabin, -PassengerId)

df_train$Age <- na.aggregate(df_train$Age)
df_train$Embarked[is.na(df_train$Embarked)] <- mfv(df_train$Embarked, na.rm=T)
df_train$TicketNbr <- replace(df_train$TicketNbr, is.na(df_train$TicketNbr), 0) %>% as.factor()

df_train %>% glimpse()

#repeat for df_test
df_test <- df_test %>% mutate(PassengerId = as.integer(PassengerId),
                              Pclass = as.factor(Pclass),
                              Sex = if_else(Sex == 'female', 1, 2) %>% as.factor(),
                              SibSp = as.ordered(SibSp),
                              Parch = as.ordered(Parch),
                              TicketNbr = str_extract(Ticket, "[:digit:]*$") %>% as.integer(),
                              Ticket = as.factor(Ticket),
                              Embarked = as.factor(Embarked),
                              Lname = str_extract(Name, "^.*(?=,)") %>% as.factor(),
                              Title = str_extract(Name, "(?<=,\\s)[:alpha:]+\\s?[:alpha:]*(?=.)") %>% as.factor()) %>%
  select(-Name, -Cabin)

df_test$Age <- na.aggregate(df_test$Age)
df_test$Embarked[is.na(df_test$Embarked)] <- mfv(df_test$Embarked, na.rm=T)
df_test$TicketNbr <- replace(df_test$TicketNbr, is.na(df_test$TicketNbr), 0) %>% as.factor()

df_test %>% glimpse()

#look for basic trends (violin plot)
y_col <- "Survived"
x_cols <- df_test %>% names()

gg_violin <- function(x_col, data){
  if(is.numeric(data[[x_col]])){
  plt <- data %>%
    ggplot(aes_string(x='Survived', y=x_col)) +
    geom_violin(draw_quantiles = c(0.25, 0.5, 0.75), fill = 'blue', alpha = 0.3, size = 1.0)+
    labs(title=str_c("Survived vs. ", x_col))

  plt %>% print()
  }
}

x_cols %>% walk(gg_violin, df_train)

#noticeable trends in the following variables
#TicketNbr, Embarked, Fare, Parch, SibSp, Age, Sex, Pclass

#look for basic trends (dotplot)
gg_dotplot <- function(x_col, data, bins = 60){
  if(is.numeric(data[[x_col]])){
  binwidth <- (max(data[,x_col]) - min(data[,x_col])) / bins
  plt <- ggplot(data, aes_string(x_col)) +
    geom_dotplot(dotsize = 0.5, method = "histodot", binwidth=binwidth) +
    facet_wrap(~Survived)+
    labs(title=str_c("Survived vs. ", x_col))

  plt %>% print()
  }
}

x_cols %>% walk(gg_dotplot, df_train)

#noticeable trends in the following variables
#TicketNbr, Embarked, Fare, Parch, SibSp, Age, Sex, Pclass

#look for basic trends (scatter plot)
gg_scatter <- function(data, x_col, y_col) {
    plt <- data %>% ggplot(mapping=aes_string(x_col, y_col))+
      geom_jitter(alpha=0.5)+
      geom_smooth(method="lm", se=FALSE)+
      coord_flip()+
      labs(title=str_c("Survived: ", x_col, " by ", y_col), subtitle="Points jittered and alpha blended")
    plt %>% print()
}

x_cols %>% walk(gg_scatter, data=df_train, y_col=y_col)

#noticeable trends in the following variables
#Title, TicketNbr, Fare, Parch, SibSp, Age, Sex, Pclass

# Remove duplicate rows
df_train <- df_train %>% distinct()
df_test <- df_test %>% distinct()

#break training set into additional training and test
set.seed(1222)

df_train2 <- df_train %>% sample_frac(0.8)
df_test2 <- df_train %>% setdiff(df_train2)

df_train2 %>% glimpse()
df_test2 %>% glimpse()

#normalize dataframes
#testing
normalize_test <- function(x)(x - mean(x, na.rm=T))/sd(x, na.rm=T)

df_test_norm <- df_test2 %>% mutate_if(is.numeric, normalize_test)

df_test_norm %>% glimpse()

#training
normalize_train <- function(col){
  mean <- mean(df_test2[[col]])
  std_dev <- sd(df_test2[[col]])
  calc <- (df_train2[[col]] - mean)/std_dev

  return(calc)
}

df_train_norm <- df_train2
df_train_norm$Age <-normalize_train("Age")
df_train_norm$Fare <-normalize_train("Fare")

df_train_norm %>% glimpse()

#statistical models
df_train_norm %>% names()

#Pclass+Sex+Age+SibSp+Parch+Ticket+Fare+Embarked+TicketNbr+Lname+Title
#Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title

#mod1
mod1 <- glm(Survived ~ Pclass+Sex+Age, family=binomial(), data=df_train2)
mod1 %>% summary()

#mod2 (broken - start here)
mod2 <- glm(Survived ~ Pclass+Sex+Age+Embarked, family=binomial(), data=df_train2)
mod2 %>% summary()

#mod3
mod3 <- glm(Survived ~ Pclass+Sex+Age+Embarked, family=binomial(), data=df_train_norm)
mod3 %>% summary()

#mod4
mod4 <- glm(Survived ~ Pclass+Sex+Age, family=binomial(), data=df_train_norm)
mod4 %>% summary()

mod1 %>% glance()
mod2 %>% glance()
mod3 %>% glance()
mod4 %>% glance()

#anova stats
anova(mod1, test="Chisq")
anova(mod2, test="Chisq")
anova(mod3, test="Chisq")
anova(mod4, test="Chisq")

#score/predict results
df_test_norm$score1 <- predict(mod1, newdata = df_test_norm)
df_test_norm$score2 <- predict(mod2, newdata = df_test_norm)
df_test_norm$score3 <- predict(mod3, newdata = df_test_norm)
df_test_norm$score4 <- predict(mod4, newdata = df_test_norm)

df_test_norm = df_test_norm %>% mutate(score1_prob = exp(score1)/(1 + exp(score1)),
                                       score2_prob = exp(score2)/(1 + exp(score2)),
                                       score3_prob = exp(score3)/(1 + exp(score3)),
                                       score4_prob = exp(score4)/(1 + exp(score4)))

df_test_norm %>% glimpse()

# dotplots for scores
score_cols <- c("score1","score1_prob","score2","score2_prob","score3","score3_prob","score4","score4_prob")

score_cols %>% walk(gg_dotplot, df_test_norm)

#violin plots for scores
score_cols %>% walk(gg_violin, df_test_norm)

#threshold functions
threshs = c(0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35)

test_threshold1 <- function(thresh){
  df_test_norm <- df_test_norm %>% mutate(score_pred1 = if_else(score1_prob > thresh, TRUE, FALSE) %>% as.factor())
  cat('For threshold of ', thresh, ' performance is: \n')
  print(caret::confusionMatrix(data=df_test_norm$score_pred1, reference = (df_test_norm$Survived %>% as.factor()), mode = "prec_recall"
  ))
  cat('\n')
}

test_threshold2 <- function(thresh){
  df_test_norm <- df_test_norm %>% mutate(score_pred2 = if_else(score2_prob > thresh, TRUE, FALSE) %>% as.factor())
  cat('For threshold of ', thresh, ' performance is: \n')
  print(caret::confusionMatrix(data=df_test_norm$score_pred2, reference = (df_test_norm$Survived %>% as.factor()), mode = "prec_recall"
  ))
  cat('\n')
}

test_threshold3 <- function(thresh){
  df_test_norm <- df_test_norm %>% mutate(score_pred3 = if_else(score3_prob > thresh, TRUE, FALSE) %>% as.factor())
  cat('For threshold of ', thresh, ' performance is: \n')
  print(caret::confusionMatrix(data=df_test_norm$score_pred3, reference = (df_test_norm$Survived %>% as.factor()), mode = "prec_recall"
  ))
  cat('\n')
}

test_threshold4 <- function(thresh){
  df_test_norm <- df_test_norm %>% mutate(score_pred4 = if_else(score4_prob > thresh, TRUE, FALSE) %>% as.factor())
  cat('For threshold of ', thresh, ' performance is: \n')
  print(caret::confusionMatrix(data=df_test_norm$score_pred4, reference = (df_test_norm$Survived %>% as.factor()), mode = "prec_recall"
  ))
  cat('\n')
}

threshs %>% walk(test_threshold1) #0.5-0.65
threshs %>% walk(test_threshold2) #0.65
threshs %>% walk(test_threshold3) #0.65
threshs %>% walk(test_threshold4) #0.6

#mod1 and mod2 are not very accurate - they were not normalized so that explains it

#find more specific threshold for mod3 and mod4
threshs = c(0.70,0.69,0.68,0.67,0.66,0.65,0.64,0.63,0.62,0.61,0.60,0.59,0.58,0.57,0.56,0.55)

threshs %>% walk(test_threshold3) #0.7+
threshs %>% walk(test_threshold4) #0.6

#find more specific threshold for mod3
threshs = c(0.75,0.74,0.73,0.72,0.71,0.70)

threshs %>% walk(test_threshold3) #0.7

#test again with better thresholds
df_test_norm <- df_test_norm %>% mutate(score_pred3 = if_else(score3_prob > 0.7, TRUE, FALSE) %>% as.factor())

cat('For threshold of ', 0.7, ' performance is: \n')
caret::confusionMatrix(data=df_test_norm$score_pred3, reference = (df_test_norm$Survived %>% as.factor()), mode = "prec_recall")

df_test_norm <- df_test_norm %>% mutate(score_pred4 = if_else(score4_prob > 0.6, TRUE, FALSE) %>% as.factor())

cat('For threshold of ', 0.6, ' performance is: \n')
caret::confusionMatrix(data=df_test_norm$score_pred4, reference = (df_test_norm$Survived %>% as.factor()), mode = "prec_recall")
#better model - higher precision and balanced accuracy

#           Reference
# Prediction FALSE TRUE
# FALSE       110   18
# TRUE          8   40
#
# Accuracy : 0.8523 
# Precision : 0.8594 
# Balanced Accuracy : 0.8109 

#McFadden R^2 index
pR2(mod3)
pR2(mod4)

#ROC curve and AUC (area under the curve)
p <- predict(mod4, newdata=subset(df_test_norm,select=(-Survived)), type="response")
pr <- prediction(p, df_test_norm$Survived)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
#0.9035652



#predict results with original df_test

#normalize df_test
normalize_final <- function(col){
  mean <- mean(df_test2[[col]])
  std_dev <- sd(df_test2[[col]])
  calc <- (df_test[[col]] - mean)/std_dev

  return(calc)
}

df_final <- df_test
df_final$Age <- normalize_final("Age")

df_final %>% glimpse()

#apply model
df_final$score <- predict(mod4, newdata = df_final)

df_final <- df_final %>% mutate(score_prob = exp(score)/(1 + exp(score)))

df_final <- df_final %>% mutate(Survived = if_else(score_prob > 0.6, TRUE, FALSE) %>% as.double(),
                                PassengerId = as.double(PassengerId)) %>% select(PassengerId,Survived)

df_final %>% glimpse()
df_final %>% return()

df_final$Survived %>% table()
df_test_norm2$Survived %>% table()

write_csv(df_final, path=str_c(path, "titanic_predictions.csv"))
