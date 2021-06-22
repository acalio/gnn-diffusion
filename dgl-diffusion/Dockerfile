FROM python:3.9-slim

ARG POETRY_VERSION=1.0.10
RUN apt-get update \
    && apt-get install build-essential cmake -y \
    && apt-get clean
RUN pip install --user  --no-cache-dir poetry==${POETRY_VERSION} 

ENV PATH="/root/.local/bin:${PATH}"

# working directory
WORKDIR /code

# create a volume to store the result
RUN mkdir /output
VOLUME /output

# initialize the poetry prokect
COPY  pyproject.toml /code/
RUN poetry run pip install --upgrade pip
RUN poetry config virtualenvs.create false
RUN poetry install 

# from this point on build-essential and cmake are no longer required
RUN apt-get purge build-essential cmake -y 

# copy everything
COPY . /code
ENTRYPOINT ["poetry", "run"]

