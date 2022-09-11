if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#**CLEAN DATA EDX & VALIDATION**

#Look at data
head(edx)
head(validation)

#split genres into distinct categories for train and test set
edx<-edx %>% separate_rows(genres, sep = "\\|")
validation<-validation %>% separate_rows(genres, sep = "\\|")

# extract Year from title for train and test set
edx <- edx %>% mutate(releaseyear = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$"))
validation <- validation %>% mutate(releaseyear = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$"))

#convert timestamp to datetime
edx$date <- as.POSIXct(edx$timestamp, origin="1970-01-01")
validation$date <- as.POSIXct(validation$timestamp, origin="1970-01-01")

#remove timestamp column
edx <- subset( edx, select = -timestamp)
validation <- subset( validation, select = -timestamp)

#get unique ratings, users, movies, genres, release year
edx %>% summarize(users = n_distinct(userId), movies = n_distinct(movieId),genres=n_distinct(genres),ratings=n_distinct(rating),movieYear=n_distinct(releaseyear))

#distribution of ratings
edx %>% 
  ggplot(aes(rating)) + 
  geom_histogram(binwidth=0.2, color="darkblue", fill="lightblue") + 
  ggtitle("Rating Distribution ")


#distribution of movies
edx %>% 
  ggplot(aes(movieId)) + 
  geom_histogram(binwidth=0.2, color="darkblue", fill="lightblue") + 
  ggtitle("Genres Distribution (Training)")

#distribution of users
edx %>% 
  ggplot(aes(userId)) + 
  geom_histogram(binwidth=0.2, color="darkblue", fill="lightblue") + 
  ggtitle("User Distribution Distribution")

#define rmse function
rmse <- function(rating, predicted){
  sqrt(mean((rating - predicted)^2))
}

#average rating for all movies
mu<-mean(edx$rating)

#predict rmse mean only model on the validation set
rmse_mean_only <- RMSE(validation$rating, mu)

#Create table to store results 
results <- data_frame(method = "Mean Only", RMSE = rmse_mean_only)

#add movie bias mean & mu
movie_mean <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

#Apply to validation set
rmse_mean_movie <- validation %>%
  left_join(movie_mean, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

#rmse for model 
mean_movie_result <- RMSE(validation$rating, rmse_mean_movie) 


# add to results table
results <- results %>% add_row(method="Movie-Based Model", RMSE=mean_movie_result)



#add user mean to the model: mean, movie, user 
user_mean <- edx %>%
  left_join(movie_mean, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#Apply to validation set
rmse_mean_movie_user<- validation %>%
  left_join(movie_mean, by='movieId') %>%
  left_join(user_mean, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#get rmse for mean_movie_user model
mean_movie_user_result <- RMSE(validation$rating, rmse_mean_movie_user)

#add result to results table
results <- results %>% add_row(method="Mean-Movie-User Model", RMSE=mean_movie_user_result)

#add genre to the model: mean, movie, user, genre
genre_mean <- edx %>%
  left_join(movie_mean, by='movieId') %>%
  left_join(user_mean, by='userId') %>%
  group_by(genres) %>%
  summarize(b_u_g = mean(rating - mu - b_i - b_u))

#Apply to validation set
rmse_mean_movie_user_genre <- validation %>%
  left_join(movie_mean, by='movieId') %>%
  left_join(user_mean, by='userId') %>%
  left_join(genre_mean, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_u_g) %>%
  pull(pred)

#get rmse for mean, movie, user, genre model
mean_movie_user_genre_result <- RMSE(validation$rating, rmse_mean_movie_user_genre)

#store results
results <- results %>% add_row(method="Mean-Movie-Use_Genre Model", RMSE=mean_movie_user_genre_result)


#REGULARIZE

#just movie
lambdas <- seq(0, 10, 0.25)

just_the_sum <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

#train model on test set
sum_r <- sapply(lambdas, function(l){
  predicted_ratings <- edx %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, edx$rating))
})
#test model on validation set
sum_r <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

#store results
results <- bind_rows(results,
                     data_frame(method="Regularized Movie",  
                                RMSE = min(sum_r)))

#regularized movie and user
rmse_mean_movie_user_r <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

#get rmse result from regularised validation set
mean_movie_user_result_r <- min(rmse_mean_movie_user_r)


#store results
results <- bind_rows(results,
                     data_frame(method="Regularized Movie - User",  
                                RMSE = min(mean_movie_user_result_r)))
#regularized mean, movie, user, genre
rmse_mean_movie_user_genre_r <- sapply(lambdas, function(lambda) {
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  b_u <- edx %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + lambda))
  
  b_u_g <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_u_g = sum(rating - b_i - mu - b_u) / (n() + lambda))
  
  
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_u_g, by='genres') %>%
    mutate(pred = mu + b_i + b_u + b_u_g) %>%
    pull(pred)
  
  
  return(RMSE(validation$rating, predicted_ratings))
})

#get rmse
mean_movie_user_genre_result_r <- min(rmse_mean_movie_user_genre_r)

#Store results
results <- bind_rows(results,
                     data_frame(method="Regularized Movie - User- Genre ",  
                                RMSE = min(mean_movie_user_genre_result_r)))

#display results
results

