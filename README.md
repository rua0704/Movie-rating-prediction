# Movie-rating-prediction
## 실행
python rating_prediction.py “training data name” “test data name”로 실행
하면 training data를 기반으로한 u#.base_prediction.txt가 나오고 PA4.exe를 사용하여 RMSE를 측
정한다.

## 구현
### Read Data
```
    data = []
    test_data = []
    f= open(sys.argv[1],'r')
    lines = f.read().split("\n")
    for line in lines:
        if not line:
            break
        data.append(line.split('\t'))
    f.close()
    df = pd.DataFrame(data,columns=["user","item","rating","timestamp"])
    df["user"]=pd.to_numeric(df["user"])
    df["item"] = pd.to_numeric(df["item"])
    df["rating"] = pd.to_numeric(df["rating"])
    df= df.drop("timestamp",axis=1)
    user_list = set(df["user"])
    item_list = set(df["item"])

    df = df.pivot(index='user', columns='item', values='rating').fillna(0.0)
```
training data를 읽어와 data에 저장해준 뒤 pandas를 이용해 DataFrame으로 만들어준다.

timestamp은 쓰지 않기 때문에 drop해준다.

user_list와 item_list는 u#.base_prediction.txt를 만들 때와 test파일에는 있고 training파일에는 없
는 new_item을 구하기 위해 사용해준다.

df를 pivot형태로 만들어준 뒤 결측값은 0으로 채워준다

### MF

```
for epoch in range(total_epoch):
        print(epoch+1," epoch")
        for u in range(rows):
            for i in range(cols):
                if r[u][i]>0.0:
                    #Stochastic Gradient Descent
                    e = r[u][i] - np.dot(p[u,:],q[:,i])
                    #loss = e**2+ lam* (np.dot(p[u,:],p[u,:].T)+np.dot(q[:,i].T,q[:,i]))
                    #if loss<min:
                    #   P=p
                    #   Q=q
                    #   min=loss
                    for j in range(k):
                        q[j][i]+=learning_rate*(e*p[u][j]-lam*q[j][i])
                        p[u][j]+=learning_rate*(e*q[j][i]-lam*p[u][j])
    #r += mean.reshape(-1, 1)
    for u in range(rows):
        for i in range(cols):
            r[u][i] = np.dot(p[u, :], q[:, i])
            if r[u][i] > 5.0:
                r[u][i] = 5.0
            elif r[u][i] < 1.0:
                r[u][i] = 1.0
```
![image](https://user-images.githubusercontent.com/68917206/135743763-9ab2852f-389c-4093-9921-17bf0cc93c7f.png)

MF의 대표적인 알고리즘 SGD(Stochastic Gradient Descent)와 ALS(Alternating Least Squares) 중에 구현이 쉽고 빠른 SGD방법을
사용했다.
## 결과
![image](https://user-images.githubusercontent.com/68917206/135743937-69c4fa8a-9f77-419c-a6b1-db7d17322765.png)

RMSE를 사용하여 결과값 확인

![image](https://user-images.githubusercontent.com/68917206/135743851-fe4f6b2b-1001-4bb1-8822-efa19ae1e8d9.png)


