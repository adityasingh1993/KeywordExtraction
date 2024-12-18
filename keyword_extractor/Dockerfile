FROM python:3.10-slim as builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
ENV PYTHONUNBUFFERED 1

# Copy only requirements to cache them in docker layer


copy models/en_core_web_sm-3.8.0-py3-none-any.whl .

RUN pip install --upgrade pip
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy
RUN pip  install sentence-transformers
RUN pip install scikit-learn
RUN pip install fastapi
RUN pip install umap-learn
RUN pip install contractions
RUN pip install fastapi==0.99.1       # Asynchronous web framework for building APIs
RUN pip install uvicorn==0.25.0        # ASGI server for running FastAPI applications
RUN pip install pandas==2.1.1          # Data manipulation and analysis library
RUN pip install numpy==1.26.4          # Fundamental package for numerical operations
RUN pip install scipy==1.11.3 
RUN pip install pydantic==1.10.13
RUN pip install seaborn
RUN pip install joblib 
RUN pip install nltk
RUN pip install tqdm  
RUN pip install en_core_web_sm-3.8.0-py3-none-any.whl --no-deps

# RUN python -m spacy download en_core_web_sm

copy .  /app
WORKDIR /app
# RUN mkdir /app/logs
#CMD ["python","-m","spacy","download","en_core_web_sm"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]