import os
from typing import List
from fastapi import FastAPI
from datetime import datetime
import pandas as pd
from catboost import CatBoostClassifier
from pydantic import BaseModel
import hashlib

# Constants for A/B testing
SALT = "recommender_salt"  # Salt for hashing
TEST_GROUP_THRESHOLD = 0.5  # 50% split between control and test groups


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


def get_model_path(model_name: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = f'/workdir/user_input/{model_name}'
    else:
        MODEL_PATH = f"/my/super/path/{model_name}"
    return MODEL_PATH


def load_models():
    model_control = CatBoostClassifier()
    model_test = CatBoostClassifier()
    model_control.load_model(get_model_path("model_control"))
    model_test.load_model(get_model_path("model_test"))
    return model_control, model_test


model_control, model_test = load_models()

# Load user and post data
users = pd.read_sql(
    """SELECT * FROM public.martynov_features_lesson_22_users """,
    con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
)

posts = pd.read_sql(
    """SELECT * FROM public.martynov_post_features_lesson_22_posts """,
    con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
)

app = FastAPI()


def get_exp_group(user_id: int) -> str:
    value_to_hash = f"{user_id}{SALT}"
    hashed_value = hashlib.md5(value_to_hash.encode()).hexdigest()
    # Используем только первые 8 символов хеша для большей равномерности
    group_value = int(hashed_value[:8], 16)
    if group_value % 2 == 0:
        return "control"
    else:
        return "test"


def recommend_posts_control(user, filtered_posts, time, limit):
    X, post_ids = prepare_features_control(user, filtered_posts, time)
    probabilities = model_control.predict_proba(X)[:, 1]
    return get_top_posts(post_ids, probabilities, limit)


def recommend_posts_test(user, filtered_posts, time, limit):
    X, post_ids = prepare_features_test(user, filtered_posts, time)
    probabilities = model_test.predict_proba(X)[:, 1]
    return get_top_posts(post_ids, probabilities, limit)


def get_top_posts(post_ids, probabilities, limit):
    predictions_df = pd.DataFrame(
        {'post_id': post_ids, 'probability': probabilities})
    top = predictions_df.sort_values(
        'probability', ascending=False).head(limit)
    extended_top = pd.merge(
        top, posts[['post_id', 'topic', 'text']], on='post_id')
    return [PostGet(id=row['post_id'], text=row['text'], topic=row['topic']) for _, row in extended_top.iterrows()]


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 5) -> Response:
    user = users[users['user_id'] == id]
    exp_group = get_exp_group(id)

    # Find user's preferred topics
    topic_columns = ['entertainment_target', 'movie_target', 'covid_target', 'sport_target',
                     'politics_target', 'business_target', 'tech_target']
    user_topics = [col for col in topic_columns if user[col].values[0] > 0]

    topic_mapping = {
        'entertainment_target': 'entertainment',
        'movie_target': 'movie',
        'covid_target': 'covid',
        'sport_target': 'sport',
        'politics_target': 'politics',
        'business_target': 'business',
        'tech_target': 'tech'
    }
    user_topic_names = [topic_mapping[topic] for topic in user_topics]

    # Filter posts based on user's preferred topics
    if len(user_topic_names) > 0:
        filtered_posts = posts[posts['topic'].isin(user_topic_names)]
    else:
        filtered_posts = posts.sort_values(
            'total_likes', ascending=False).head(50)

    # Apply the appropriate model based on the experimental group
    if exp_group == 'control':
        recommendations = recommend_posts_control(
            user, filtered_posts, time, limit)
        print(f"User {id} in control group, using control model")
    elif exp_group == 'test':
        recommendations = recommend_posts_test(
            user, filtered_posts, time, limit)
        print(f"User {id} in test group, using test model")
    else:
        raise ValueError('Unknown group')

    return Response(exp_group=exp_group, recommendations=recommendations)


def prepare_features_control(user, posts, timestamp: datetime):
    user_features_series = user.iloc[0]
    posts_with_user_features = posts.assign(**user_features_series)

    day_of_week = timestamp.weekday()
    hour = timestamp.hour
    time_slot = f"{day_of_week}_{hour}"

    posts_with_user_features['time_slot'] = time_slot
    post_ids = posts_with_user_features['post_id'].copy()
    posts_with_user_features = posts_with_user_features.drop(
        ['user_id', 'text', 'post_id'], axis=1)

    # Исключаем столбцы с "embedding" в названии
    columns_to_keep = [
        col for col in posts_with_user_features.columns if 'embedding' not in col.lower()]
    features_for_prediction = posts_with_user_features[columns_to_keep].reindex(
        columns=model_control.feature_names_)

    return features_for_prediction, post_ids


def prepare_features_test(user, posts, timestamp: datetime):
    user_features_series = user.iloc[0]
    posts_with_user_features = posts.assign(**user_features_series)

    day_of_week = timestamp.weekday()
    hour = timestamp.hour
    time_slot = f"{day_of_week}_{hour}"

    posts_with_user_features['time_slot'] = time_slot
    post_ids = posts_with_user_features['post_id'].copy()
    posts_with_user_features = posts_with_user_features.drop(
        ['user_id', 'text', 'post_id'], axis=1)

    # Используем все столбцы для тестовой модели
    features_for_prediction = posts_with_user_features.reindex(
        columns=model_test.feature_names_)

    return features_for_prediction, post_ids
