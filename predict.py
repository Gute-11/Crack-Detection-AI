import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model_path = 'runs/segment/train2/weights/best.pt'

model = YOLO(model_path)

#img_path = 'C:/Users/Gugu/Projeto_IA_Rachadura/361_teste.JPG'
img_path = 'C:/Users/Gugu/Projeto_IA_Rachadura/rachadura_net_2.JPG'
#img_path = 'C:/Users/Gugu/Projeto_IA_Rachadura/IMAGEM_RACHADURA_TESTE_PERFEITO.JPG'
#img_path = 'C:/Users/Gugu/Projeto_IA_Rachadura/rachadura-inclinada-parede.png'

results = model(img_path, task='segment')

result = results[0]

result.show()

result.save()

if result.masks:
    
    for mask in result.masks:
        plt.imshow(mask.cpu().numpy(), cmap='gray')
        plt.show()

else:
    print("Nenhuma máscara de segmentação foi encontrada.")
