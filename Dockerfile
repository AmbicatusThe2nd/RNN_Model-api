FROM python:3.8-slim-buster
RUN pip3 install flask
RUN pip3 install pandas
RUN pip3 install scikit-learn==1.1.2
COPY . .
EXPOSE 5000
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]