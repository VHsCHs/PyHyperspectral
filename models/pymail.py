# coding:utf-8
# smtplib模块负责连接服务器和发送邮件
# MIMEText：定义邮件的文字数据
# MIMEImage：定义邮件的图片数据
# MIMEMultipart：负责将文字图片音频组装在一起添加附件
import os, sys, math
import smtplib  # 加载smtplib模块
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


class MAIL():
    def __init__(self, username='xxx@126.com', passwd='xxx', mailserver='smtp.126.com', port=25):
        self.username = username  # 'xxx@126.com' 发件人邮箱账号
        self.passwd = passwd  # 'xxx'
        self.mailserver = mailserver  # 'smtp.126.com'
        self.port = port  # '25'

    def send(self, receive='receive@receive.com', subtitle='subtitle', body='body', attachment=False, img=False):
        try:
            msg = MIMEMultipart('related')
            msg['From'] = formataddr(["sender", self.username])  # 发件人邮箱昵称、发件人邮箱账号
            msg['To'] = formataddr(["receiver", receive])  # 收件人邮箱昵称、收件人邮箱账号
            msg['Subject'] = subtitle

            # 文本信息
            text = MIMEText(body, 'plain', 'utf-8')
            msg.attach(text)

            if attachment != False:
                # 附件信息
                attach = MIMEApplication(open(attachment).read())
                attach.add_header('Content-Disposition', 'attachment', filename=os.path.split(attachment)[1])
                msg.attach(attach)

            if img != False:
                # 正文显示图片
                body = """
                <b>this is a test mail:</b> 
                <br><img src="cid:image"><br>
                """
                text = MIMEText(body, 'html', 'utf-8')
                with open(img, 'rb') as f:
                    pic = MIMEImage(f.read())
                pic.add_header('Content-ID', '<image>')
                msg.attach(text)
                msg.attach(pic)

            server = smtplib.SMTP_SSL(self.mailserver, self.port)  # 发件人邮箱中的SMTP服务器，端口是25
            server.login(self.username, self.passwd)  # 发件人邮箱账号、邮箱密码
            server.sendmail(self.username, receive, msg.as_string())  # 发件人邮箱账号、收件人邮箱账号、发送邮件
            server.quit()
            print('success')
        except Exception as ErrorReport:
            print(ErrorReport)


if __name__ == '__main__':
    mail = MAIL(username='111@qq.com', passwd='111', mailserver='smtp.qq.com', port=465)
    mail.send(receive='111@qq.com', subtitle=os.path.basename(sys.argv[0]).split(".")[0],
              body='Done!')
