
git clone -b gradio https://github.com/tvolk19/cosmos-predict2.git && cd cosmos-predict2
sed -i -e 's/h11==0.16.0/h11==0.14.0/g' /etc/pip/constraint.txt
python3 -m pip install gradio --break-system-packages
PYTHONPATH=$(pwd) python server/gradio_bootstrapper.py
