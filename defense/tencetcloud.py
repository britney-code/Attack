
import json
import types
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tiia.v20190529 import tiia_client, models
import base64
import os 
attack_success = 0 
num = 1000
# 实例化一个认证对象，入参需要传入腾讯云账户 SecretId 和 SecretKey，此处还需注意密钥对的保密
# 代码泄露可能会导致 SecretId 和 SecretKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议采用更安全的方式来使用密钥，请参见：https://cloud.tencent.com/document/product/1278/85305
# 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取
cred = credential.Credential("", "")
# # 实例化一个http选项，可选的，没有特殊需求可以跳过
httpProfile = HttpProfile()
httpProfile.endpoint = "tiia.tencentcloudapi.com"
# # 实例化一个client选项，可选的，没有特殊需求可以跳过
clientProfile = ClientProfile()
clientProfile.httpProfile = httpProfile
# # 实例化要请求产品的client对象,clientProfile是可选的
client = tiia_client.TiiaClient(cred, "ap-beijing", clientProfile)
# 实例化一个请求对象,每个接口都会对应一个request对象
req = models.DetectLabelProRequest()

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            # 将图片内容转换为 Base64 编码，并解码为 UTF-8 字符串
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_image
    except Exception as e:
        print(f"Error converting image to Base64: {e}")
        return None
    
def save_labels_to_file(labels, filename):
    with open(filename, "w", encoding="utf-8") as file:
        for label in labels:
            file.write(label + "\n")

def load_labels(filename): 
    labels = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file: 
            labels.append(line.strip())
    return labels 

# data_folder_path = './data/images/'
# # 获取图像文件列表
# image_files = [f for f in os.listdir(data_folder_path) if f.endswith('.png')]
# correct_label = []
# j = 0 

# # 记录 NULL 值
# nullIndex = []
# for image_file in image_files: 
#     image_path = os.path.join(data_folder_path, image_file)
#     encoded_image = image_to_base64(image_path) 
#     params = {"ImageBase64": encoded_image} 
#     req.from_json_string(json.dumps(params))
#     # 返回的resp是一个DetectLabelProResponse的实例，与请求对象对应
#     resp = client.DetectLabelPro(req).to_json_string()
#     # # 输出json格式的字符串回包
#     resp = json.loads(resp)["Labels"]
#     if resp:
#         top1 = max(resp, key = lambda x: x["Confidence"])
#         label = top1["Name"]
#         Confidence = top1["Confidence"]
#         print(f"第{j}个原始样本top1:{label} \t\t\t top1置信度:{Confidence}")
#         correct_label.append(label)
#     else: 
#         print(f"预测缺失！")
#         correct_label.append("NULL")
#         nullIndex.append(str(j)) # 记录为空的j值
#     j += 1 
#     if j == num: break 

# save_labels_to_file(correct_label, "tencetAIcloud_correct_label.txt")
# save_labels_to_file(nullIndex, "NULL_Index.txt")

# 加载 
load_correct_label = load_labels("tencetAIcloud_correct_label.txt")
load_null_index = load_labels("NULL_Index.txt")

# # # 对抗样本 
adv_folder_path = r"./result/pamens" 
adv_image_files = [f for f in os.listdir(adv_folder_path) if f.endswith('.png')]
i = 0
for adv in adv_image_files:
    if load_correct_label[i] == "NULL": # 如果当前为NULL，那么直接跳过，判断下一次i
        print(f"第{i}个对抗样本缺少真实标签,不判断！")
        i += 1 
        continue 
    adv_path = os.path.join(adv_folder_path, adv)
    encoded_image = image_to_base64(adv_path) 
    params = {"ImageBase64": encoded_image} 
    req.from_json_string(json.dumps(params))
    # 返回的resp是一个DetectLabelProResponse的实例，与请求对象对应
    resp = client.DetectLabelPro(req).to_json_string()
    resp = json.loads(resp)["Labels"]
    if resp:
        adv_top1 = max(resp, key = lambda x: x["Confidence"])
        adv_label = adv_top1["Name"]
        adv_Confidence = adv_top1["Confidence"]
        print(f"第{i}个对抗样本top1:{adv_label} \t\t\t top1置信度:{adv_Confidence}")
        if adv_label != load_correct_label[i]: 
           attack_success += 1 
    else: 
        print(f"第{i}个对抗样本攻击成功！")
        attack_success += 1 
    i += 1 
    if i == num: break 
    
length = num - len(load_null_index)
print(f"腾讯云攻击成功率为:{(attack_success / length) * 100}%")
        
    