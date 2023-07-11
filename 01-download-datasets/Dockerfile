FROM debian:buster-slim
RUN apt-get update \
&& apt-get install -y python3-pip python3-dev python3-numpy default-jdk \
&& cd /usr/local/bin \
&& ln -s /usr/bin/python3 python 
COPY ./crawler-inema /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["bash","run.sh"]
