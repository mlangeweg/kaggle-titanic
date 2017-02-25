# kaggle
# titanic competition using random forests

library(dplyr)
library(mice)
library(randomForest)
library(ggplot2)

#set working directory

#read in data
train <- read.csv("../titanic/train.csv", stringsAsFactors = FALSE)
test <- read.csv("../titanic/test.csv", stringsAsFactors = FALSE)

#combine data sets for feature engineering
test$Survived <- NA
data <- bind_rows(train, test)
data$Name <- as.character(data$Name)

#pull title from names
data$Title <- gsub('(.*, )|(\\..*)', '', data$Name)

#group rare titles
u_title <- c("Dona", "Lady", "the Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer")

data$Title[data$Title == 'Mlle']        <- 'Miss' 
data$Title[data$Title == 'Ms']          <- 'Miss'
data$Title[data$Title == 'Mme']         <- 'Mrs' 
data$Title[data$Title %in% u_title]  <- 'Rare Title'

#pull surname from passengers
data$Surname <- sapply(data$Name, function(x) strsplit(x, split = "[,.]")[[1]][1])

# use median of "Fare" to compute missing values
data$Fare[is.na(data$Fare)] <- median(data$Fare, na.rm=TRUE)

#replace missing values of embarked
data$Embarked[data$Embarked==""] = "S"

set.seed(1)
#use mice algorithm to compute missing values in "Age" feature
imputed <- mice(data[, !names(data) %in% c("PassengerId", "Survived", "Name", "Cabin", "Ticket", "Surname")])
mice_out <- complete(imputed)
data$Age <- mice_out$Age

#split data into train and test
train01 <- data[1:891,]
test01 <- data[892:1309,]

# convert select features to factor variables
train01$Sex <- as.factor(train01$Sex)
train01$Embarked <- as.factor(train01$Embarked)
train01$Title <- as.factor(train01$Title)

test01$Sex <- as.factor(test01$Sex)
test01$Embarked <- as.factor(test01$Embarked)
test01$Title <- as.factor(test01$Title)

set.seed(2)
#construct random forest model
rf <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title,
                                                data = train01,
                                                importance = TRUE,
                                                ntree = 2000)

#create submission file with predictions
submission <- data.frame(PassengerId = test01$PassengerId)
submission$Survived <- predict(rf, test01)
write.csv(submission, file = "01_submission.csv", row.names=FALSE)

# compute importance of features
imp <- importance(rf, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

#construct plot of feature importance
p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
     geom_bar(stat="identity", fill="#53cfff") +
     coord_flip() + 
     theme_light(base_size=20) +
     xlab("") +
     ylab("Importance") + 
     ggtitle("Random Forest Feature Importance\n") +
     theme(plot.title=element_text(size=18))

ggsave("feature_importance.png", p)
