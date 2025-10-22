import pickle

if __name__ == '__main__':
    model_lr = pickle.load(open("log_reg.pkl", "rb"))

    can_uoc_tinh = [[50.050049,52.082081,48.028027,50.220219]]
    gia_tri_du_doan= model_lr.predict(can_uoc_tinh)
    print(f"Gia tri du doan: {gia_tri_du_doan}")