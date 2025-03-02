# from aip import AipImageClassify
#
# """ 你的 APPID AK SK """
# APP_ID = ''
# API_KEY = ''
# SECRET_KEY = ''
#
# client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)
#
# """ 读取图片 """
#
#
# def get_file_content(filePath):
#     with open(filePath, 'rb') as fp:
#         return fp.read()
#
#
# image = get_file_content('./dataset/images/0aebe24fc257286e.png')
#
# # """ 调用通用物体和场景识别 """
# # result = client.advancedGeneral(image)
# # print(result)
#
# """ 如果有可选参数 """
# options = {"top_num": "1"}
# options["top_num"] = 1
# """ 带参数调用通用物体和场景识别 """
# result = client.advancedGeneral(image, options=options)
# print(result)
from aip import AipImageClassify
import os

# 替换为你的 APPID AK SK
APP_ID = ''
API_KEY = ''
SECRET_KEY = ''
client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)
# 读取图片文件
def get_file_content(file_path):
    with open(file_path, 'rb') as fp:
        return fp.read()

attacksuccess = 0  # 攻击成功的图像
num = 1000
correct_label = []
# 图像文件夹路径
image_folder_path = './data/images/'
# 获取图像文件列表
image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.png')]
print(image_files)

j = 0
# 循环处理每张图像
for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    image_content = get_file_content(image_path)
    # 带参数调用通用物体和场景识别
    result_with_options = client.advancedGeneral(image_content, options={"result_num": 1})
    # print(f"带参数识别结果 for {image_file}: {result_with_options}")
    result_top1 = result_with_options["result"][0]["keyword"]
    result_top1_score = result_with_options["result"][0]["score"]
    print(f"第{j}个原始样本预测为:  {result_top1}")
    correct_label.append(result_top1)
    j += 1
    if j == 1000: break


with open("correct_label.txt", "w") as file:
    for label in correct_label:
         file.write(label + "\n")


# 从txt文件加载为列表
loaded_correct_label = []
with open("correct_label.txt", "r") as file:
    for line in file:
        loaded_correct_label.append(line.strip())
print(loaded_correct_label)
# 对抗样本图像的文件夹
adv_folder_path = r"./result/apens"
adv_image_files = [f for f in os.listdir(adv_folder_path) if f.endswith('.png')]
print(adv_image_files)
i = 0
for adv in adv_image_files:
    adv_path = os.path.join(adv_folder_path, adv)
    adv_content = get_file_content(adv_path)
    # 带参数调用通用物体和场景识别
    result_with_options = client.advancedGeneral(adv_content, options={"result_num": 1})
    result_top1_adv_label = result_with_options["result"][0]["keyword"]
    print(f"第{i}个对抗样本预测为:  {result_top1_adv_label}")
    if result_top1_adv_label != loaded_correct_label[i]:
        attacksuccess += 1
    i += 1
    if i == 1000: break
print(f"百度云攻击成功率为:{(attacksuccess / num) * 100}%")