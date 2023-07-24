FROM debian
RUN apt-get update \
&& apt-get install -y python3-pip python3-dev python3-numpy default-jdk git \
&& cd /usr/local/bin \
&& ln -s /usr/bin/python3 python 
COPY . /app
WORKDIR /app
RUN pip3 install --break-system-packages -r requirements.txt
ENTRYPOINT ["bash","run.sh"]