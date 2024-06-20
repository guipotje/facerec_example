FROM ultralytics/ultralytics:8.2.36-cpu

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the application source code into the container
#COPY ./src /usr/src/app

# Set an environment variable if needed (example provided)
# ENV MY_ENV_VAR=my_value

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV OMP_NUM_THREADS=4

# Expose a port if your application needs it
EXPOSE 8080