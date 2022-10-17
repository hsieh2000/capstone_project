import yaml
import re
import json

config_path = "../config.json"

with open(config_path) as j:
    config = json.load(j)
    len_id = config['len']

main_data_path = "../data/face_model_main.yml"
new_data_path =  f"../data/face_model_{len_id}.yml"
export_path = "../data/face_model_main.yml"
config_path = "../config.json"

with open(new_data_path, "r") as f:
    x = f.read()
with open(main_data_path, "r") as g:
    y = g.read()

class yaml(object):
    def __init__(self):
        a = self.preprossing(x, y)
        a = self.export(a)

    def preprossing(self, x, y):

        x_data = x.split("histograms:")[1].split("\n   labels: !!opencv-matrix")[0]
        y_data = y.split('\n   labels: !!opencv-matrix')[0]
        x_label = x.split("\n   labels: !!opencv-matrix")[1].split('data: [')[1].split("]\n   labelsInfo:\n")[0]
        y_label =y.split('\n   labels: !!opencv-matrix')[1].split("data: [")[1].split(" ]\n   labelsInfo:\n")[0]
        new_data = y_data+x_data

        regex = re.compile("\d{1,2}")
        x_label_amount = re.findall(regex, x_label)
        y_label_amount = re.findall(regex, y_label)

        new_label_amount =y_label_amount+x_label_amount
        len_id = sorted(list(set(new_label_amount)), reverse = False)[-1]
        self.config(len_id)

        label_tem ='\n   labels: !!opencv-matrix\n      rows: ddd\n      cols: 1\n      dt: i\n      data: [xxx]\n   labelsInfo:\n      []\n '
        label_tem = label_tem.replace('ddd', str(len(new_label_amount)))

        text=""
        for i, j in enumerate(new_label_amount):
            if (i+1)%20==0 and i+1!=len(new_label_amount):
                text = text+f" {j},\n         "

            elif i+1 == len(new_label_amount):
                text = text+f" {j} "

            else:
                text = text+f" {j},"

        new_label =label_tem.replace("xxx", text)
        new = new_data+new_label

        return new

    def export(self, model):
        with open(export_path, "w") as g:
            g.write(model)

    def config(self, len_id):
        with open(config_path) as j:
            config = json.load(j)
            dict_ = config
            dict_['len'] = len_id
            
        with open(config_path, 'w') as j:
            json.dump(dict_, j)

if __name__ == "__main__":
    c = yaml()