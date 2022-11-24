import web

web.config.debug = False
from web import form

import librosa.display
import numpy as np
import pandas as pd
import os
from tensorflow import keras
import h5py
from sklearn.utils import shuffle
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# 对应请求，采用对应类来处理
urls = (
    '/', 'index',
    '/index2', 'index2',
    '/ins2', 'ins2',
    '/register', 'register',
    '/login', 'login',
    '/logout', 'logout',
    '/csc', 'csc',
)
# 创建一个web应用处理用户的请求
app = web.application(urls, globals())

render = web.template.render('templates/')  # 用来渲染放在templates目录文件夹下的网页
# 创建一个数据库对象，web.py可自动处理与数据库的链接和断开
db = web.database(dbn='mysql', host="gz-cdb-qpc11qjx.sql.tencentcdb.com", port=56907
                  , user="root", passwd="123456aa", database="bysj")

session = web.session.Session(app, web.session.DiskStore('sessions'), initializer={'username': None})

# 登录表单form模板
loginform = form.Form(
    form.Textbox('username',
                 form.notnull,
                 # form.regexp('[A-Za-z0-9\-]+', '必须是字母或数字！'),
                 # form.Validator('必须超过5个字符！', lambda y: len(y) > 5)
                 ),
    form.Password('password',
                  form.notnull,
                  #  form.regexp('[A-Za-z0-9\-]+', '必须是字母或数字！'),
                  # form.Validator('必须超过5个字符！', lambda y: len(y) > 5)
                  ),
    form.Button('登录')
)


class index:
    def GET(self):
        # 确保通过调用表单来创建表单的副本 (上方的线条)
        # 否则，将全局显示更改
        if not session.username:
            raise web.seeother('/login')
        else:
            username = session.username
            return render.index(username)


class index2:
    def GET(self):
        if not session.username:
            raise web.seeother('/login')
        else:
            os.system("python yy.py")
            username = session.username
            return render.index2(username)

    # def POST(self):
    #     b = web.input()
    #     if b.a:
    #         os.system("python yy.py")
    #         raise web.seeother('/')


class ins2:
    def GET(self):
        if not session.username:
            raise web.seeother('/login')
        else:
            feeling_list = []
            # 所有数据
            mylist = os.listdir('data/')
            # 遍历数据
            for item in mylist:
                if item[6:-16] == '02' and int(item[18:-4]) % 2 == 0:
                    feeling_list.append('女性平静')  # 女性平静 female_calm
                elif item[6:-16] == '02' and int(item[18:-4]) % 2 == 1:
                    feeling_list.append('男性平静')  # 男性平静
                elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 0:
                    feeling_list.append('女性开心')  # 女性开心
                elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 1:
                    feeling_list.append('男性开心')  # 男性开心
                elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 0:
                    feeling_list.append('女性悲伤')  # 女性悲伤
                elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 1:
                    feeling_list.append('男性悲伤')  # 男性悲伤
                elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 0:
                    feeling_list.append('女性愤怒')  # 女性愤怒
                elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 1:
                    feeling_list.append('男性愤怒')  # 男性愤怒
                elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 0:
                    feeling_list.append('女性恐惧')  # 女性恐惧
                elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 1:
                    feeling_list.append('男性恐惧')  # 男性恐惧
                elif item[:1] == 'a':
                    feeling_list.append('男性愤怒')  # 男性愤怒
                elif item[:1] == 'f':
                    feeling_list.append('男性恐惧')  # 男性恐惧
                elif item[:1] == 'h':
                    feeling_list.append('男性开心')  # 男性开心
            # elif item[:1]=='n':
            # feeling_list.append('neutral')
                elif item[:2] == 'sa':
                    feeling_list.append('男性悲伤')  # 男性悲伤

                # 构建label Dataframe
        labels = pd.DataFrame(feeling_list)

        # 构建1个包含feature特征列的Dataframe
        df = pd.DataFrame(columns=['feature'])
        bookmark = 0
        # 遍历数据
        for index, y in enumerate(mylist):
            if mylist[index][6:-16] not in ['01', '07', '08'] and mylist[index][:2] != 'su' and mylist[index][
                                                                                                :1] not in ['n',
                                                                                                            'd']:
                X ,sample_rate = librosa.load('data/' + y,
                                              res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
                mfccs = librosa.feature.mfcc(y=X, sr=np.array(sample_rate), n_mfcc=13)
                feature = np.mean(mfccs, axis=0)
                df.loc[bookmark] = [feature]
                bookmark = bookmark + 1
                # 拼接特征与标签
        df3 = pd.DataFrame(df['feature'].values.tolist())
        newdf = pd.concat([df3, labels], axis=1)
        # 重命名标签字段
        rnewdf = newdf.rename(index=str, columns={"0": "label"})
        # 打乱样本顺序
        rnewdf = shuffle(newdf)
        # 80%的训练集，20%的测试集
        newdf1 = np.random.rand(len(rnewdf)) < 0.8
        train = rnewdf[newdf1]
        test = rnewdf[~newdf1]
        # 训练集特征与标签
        trainfeatures = train.iloc[:, :-1]
        trainlabel = train.iloc[:, -1:]
        # 测试集特征与标签
        testfeatures = test.iloc[:, :-1]
        testlabel = test.iloc[:, -1:]
        # 转为numpy array格式
        X_train = np.array(trainfeatures)
        y_train = np.array(trainlabel)
        X_test = np.array(testfeatures)
        y_test = np.array(testlabel)
        # 映射编码
        lb = LabelEncoder()
        y_train = to_categorical(lb.fit_transform(y_train))
        y_test = to_categorical(lb.fit_transform(y_test))
        # 模型重加载与测试集评估
        model_path = h5py.File(
            'model/Emotion_Voice_Detection_Model.h5',
            'r')
        loaded_model = keras.models.load_model(model_path)
        X, sample_rate = librosa.load('output/output12.wav',
                                      res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=np.array(sample_rate), n_mfcc=13), axis=0)
        livedf = pd.DataFrame(data=mfccs)
        livedf = np.expand_dims(livedf.stack().to_frame().T, axis=2)
        livepreds = loaded_model.predict(livedf, batch_size=32, verbose=1)
        c = lb.inverse_transform(livepreds.argmax(axis=1))
        cc = c
        username = session.username
        return render.ins2(cc, username)


class register:
    def GET(self):
        return render.register()

    def POST(self):
        i = web.input()
        if i.username:
            userInsert = db.insert('user', username=i.username, password=i.pwd1)
            raise web.seeother('/login')
        else:
            return render.register()


class login:
    def GET(self):
        form = loginform()
        # 确保通过调用表单来创建表单的副本 (上方的线条)
        # 否则，将全局显示更改
        return render.login(form, user='user')

    def POST(self):
        form = loginform()
        if not form.validates():
            return render.login(form, user='user')
        else:
            users = db.query('select * from user where username=$username', vars={'username': form.d.username})
            result = users[0]  # None

            if result and result.password == form.d.password:
                session.username = form.d.username
                raise web.seeother('/')
            return render.login(form, user=None)


class logout:
    def GET(self):
        session.username = None
        raise web.seeother('/csc')


class csc:
    def GET(self):
        return render.csc()


# 要想python文件在终端执行就必须有这个if, __name__（文件自带的属性）只有文件在终端执行时才会变成__main__
if __name__ == "__main__":
    # 启动这个应用
    app.run()
