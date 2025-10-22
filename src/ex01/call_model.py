import pickle

if __name__ == '__main__':
    model_lr = pickle.load(open("model1.pkl", "rb"))

    can_uoc_tinh = [[0.85204,0,8.14,0,0.538,5.965,89.2,4.0123,4,307,21,13.83]]
    gia_tri_du_doan= model_lr.predict(can_uoc_tinh)
    print(f"Gia tri du doan: {gia_tri_du_doan}")