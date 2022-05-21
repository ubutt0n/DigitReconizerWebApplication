FROM jjanzic/docker-python3-opencv

ADD digitrec.py .
ADD model.pth .

RUN pip install streamlit torch pandas numpy

CMD ["streamlit", "run", "./digitrec.py"]