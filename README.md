# 🚀 Post Recommendation System

A powerful recommendation system that selects the top 5 most relevant posts for a user based on their interaction history and preferences.

## 📊 Data Sources

Our system utilizes three main tables from PostgreSQL:

- **📋 user_data**: Comprehensive user information
- **📝 post_text**: Detailed post content and metadata
- **🔄 feed_data**: User-post interaction history

## 🏗️ Project Structure

The project consists of three main components:

1. **🚀 API**: A FastAPI server with a single GET endpoint

   - Send a user ID in the URL
   - Receive 5 most relevant posts
   - Implements A/B testing with control and test groups

2. **🔍 Feature Extraction**: `preload_features.ipynb`

   - Extracts features from user and post data
   - Saves preprocessed features in PostgreSQL
   - Converts post text to embeddings using `RobertaModel`

3. **🧠 Model Training**: `train.ipynb`
   - Trains two CatBoost models (control and test) using preprocessed features
   - Saves the models for use in the API

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFB13B?style=for-the-badge&logo=catboost&logoColor=black)

### Main Libraries

| Library       | Version | Description                                   |
| ------------- | ------- | --------------------------------------------- |
| FastAPI       | 0.75.1  | High-performance web framework                |
| SQLAlchemy    | 1.4.35  | SQL toolkit and ORM                           |
| CatBoost      | 1.0.6   | Gradient boosting library                     |
| Pandas        | 1.4.2   | Data manipulation and analysis                |
| NumPy         | 1.22.4  | Numerical computing tools                     |
| Pydantic      | 1.9.1   | Data validation using Python type annotations |
| Psycopg2      | 2.9.3   | PostgreSQL adapter for Python                 |
| python-dotenv | 0.19.1  | Reading key-value pairs from .env file        |

## 🚀 Getting Started

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/post-recommendation-system.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   Create a `.env` file in the root directory with the following content:

   ```
   DB_CONNECTION_STRING=postgresql://username:password@host:port/database
   ```

4. Run the feature extraction notebook:

   ```
   jupyter notebook preload_features.ipynb
   ```

5. Train the models:

   ```
   jupyter notebook train.ipynb
   ```

6. Start the API server:
   ```
   uvicorn api:app --reload
   ```

## 🎯 API Usage

To get recommendations for a user:

```
GET /post/recommendations/?id={user_id}&time={timestamp}&limit={limit}
```

Parameters:

- `id`: User ID (required)
- `time`: Current timestamp (required)
- `limit`: Number of recommendations to return (optional, default is 5)

Response:

```json
{
  "exp_group": "control",
  "recommendations": [
    {"id": 1234, "text": "Post content...", "topic": "tech"},
    {"id": 5678, "text": "Another post...", "topic": "entertainment"},
    ...
  ]
}
```

## 🧪 A/B Testing

The system implements A/B testing:

- Users are deterministically assigned to either the "control" or "test" group.
- Different models are used for each group to generate recommendations.
- The `exp_group` field in the response indicates which group the user belongs to.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/post-recommendation-system/issues).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
