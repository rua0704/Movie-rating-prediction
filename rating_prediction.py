import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
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

    f = open(sys.argv[2], 'r')
    lines = f.read().split("\n")
    for line in lines:
        if not line:
            break
        test_data.append(line.split('\t'))
    f.close()
    df_test = pd.DataFrame(test_data, columns=["user", "item", "rating", "timestamp"])
    df_test["user"] = pd.to_numeric(df_test["user"])
    df_test["item"] = pd.to_numeric(df_test["item"])
    user_test_list = set(df_test["user"])
    item_test_list = set(df_test["item"])
    new_item = item_test_list - item_list
    new_user = user_test_list-user_list

    total_epoch = 100
    learning_rate = 0.0005
    lam = 0.004
    k = 4

    r = df.values
    #mean = np.mean(r,axis=1)
    #r -=mean.reshape(-1,1)
    rows = len(r)
    cols = len(r[0])
    p = np.random.rand(rows,k)
    q = np.random.rand(k,cols)

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

    f = open(sys.argv[1]+"_prediction.txt", 'w')
    user = 0
    user_mean = r.mean(axis=1)
    for i in user_list:
        item = 0
        for j in item_list:
            f.write(str(i) + "\t" + str(j) + "\t" + str(r[user, item ]) + "\n")
            item +=1

        for k in new_item:
            f.write(str(i)+"\t"+str(k)+"\t"+ str(user_mean[user])+"\n")
        user += 1
    f.close()