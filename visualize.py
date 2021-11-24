import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# csv = pd.read_csv("C:\\Users\\home\\Desktop\\csv\\tta0.01.csv")
# csv = pd.read_csv("C:\\Users\\home\\Desktop\\csv\\yolov5x320tta.csv")
csv = pd.read_csv("C:\\Users\\home\\Desktop\\csv\\tta1130.csv")

# csv = csv[csv["confidence"] >= 0.3247]

print(csv[csv["class_id"] == 1]["confidence"].describe())
print(csv[csv["class_id"] == 2]["confidence"].describe())
print(csv[csv["class_id"] == 3]["confidence"].describe())
print(csv[csv["class_id"] == 4]["confidence"].describe())


print(csv[csv["class_id"] == 1]["confidence"].count() / csv["confidence"].count() * 100)
print(csv[csv["class_id"] == 2]["confidence"].count() / csv["confidence"].count() * 100)
print(csv[csv["class_id"] == 3]["confidence"].count() / csv["confidence"].count() * 100)
print(csv[csv["class_id"] == 4]["confidence"].count() / csv["confidence"].count() * 100)

csv.to_csv("baseline.csv", index=False)

#
# csv = csv[csv["confidence"] >= 0.1]
#
# # print(csv)
# csv = csv.sort_values(by='confidence', ascending=True)
#
# conf1 = csv[csv["class_id"] == 1]['confidence']
# conf2 = csv[csv["class_id"] == 2]['confidence']
# conf3 = csv[csv["class_id"] == 3]['confidence']
# conf4 = csv[csv["class_id"] == 4]['confidence']
#
#
# sb.kdeplot(conf1)
# plt.title("confidence1")
# plt.show()
#
# sb.kdeplot(conf2)
# plt.title("confidence2")
# plt.show()
#
# sb.kdeplot(conf3)
# plt.title("confidence3")
# plt.show()
#
# sb.kdeplot(conf4)
# plt.title("confidence4")
# plt.show()
