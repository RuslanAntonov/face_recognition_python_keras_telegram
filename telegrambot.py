import cv2
import re
import telethon
import os
from telethon import TelegramClient, events, sync

api_id = #
api_hash = ''
phone = ""
chat = ""

client = TelegramClient('coma1', api_id, api_hash)
client.start()
client.sign_in(phone)

def send_screen(img):
        client.send_message(chat, 'Обнаружено неустановленное лицо')
        cv2.imwrite('screen.jpg',img)
        client.send_file(chat,'screen.jpg')
        os.remove('screen.jpg')
