import logging
import os

from django.core.mail import EmailMultiAlternatives
from web.settings import EMAIL_HOST_USER, MEDIA_URL
from django.template.loader import get_template

import requests

logger = logging.getLogger(__name__)


def send_email(video_path, time_fall, phone_number):
    time_fall = str(time_fall)
    video_link = MEDIA_URL + str(video_path)
    subject, from_email, to = 'Система детектирования падений', EMAIL_HOST_USER, EMAIL_HOST_USER
    text_content = 'This is an important message.'
    # mail_html_path = os.path.dirname(__file__) + "email.html"
    html_ = get_template('email.html')
    context_ = {
        'time_fall': time_fall,
        'video_link': video_link,
        'phone_number': phone_number
    }
    html_content = html_.render(context_)
    print(to)
    print(EMAIL_HOST_USER)
    print(from_email)
    print(subject)

    def send_msg(text):
        token = "5624864341:AAFQd3nWe8uinePbq3MKMPfbrdi7yDB5ax8"
        chat_id = "@public_link_a"
        url_req = "https://api.telegram.org/bot" + token + "/sendMessage" + "?chat_id=" + chat_id + "&text=" + text
        results = requests.get(url_req)
        print(results.json())

    send_msg(f"❗️Внимание! Произошло падение\nВремя:{time_fall}\n")

    # msg = EmailMultiAlternatives(subject, text_content, from_email, [to, EMAIL_HOST_USER])
    # msg.attach_alternative(html_content, "text/html")
    # msg.send()

# send_msg(f"❗️Внимание! Произошло падение\nВремя:11:35\n")