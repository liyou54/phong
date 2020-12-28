import taichi as ti
import numpy as np
import matplotlib.image as mpig

ti.init(ti.gpu)
gui = ti.GUI()
PAI = 3.1415926535897932384
# 布林冯模型
# 使用左手坐标系
#    y上 z前
#     | /   
#     0 — x右
WIDGHT, HEIGTH = 512, 512
LIGHT_POWER = 1.0
ColorBase = 0.2
model_data = ti.Vector(4, dt=ti.f64, shape=8)
tri_index = ti.Vector(3, dt=int, shape=12)
LIGHT_POSITION = ti.Vector([1, -0.5, 1])
#  深度索引
deep_index = ti.field(dtype=int, shape=(WIDGHT, HEIGTH))
res_pic = ti.Vector(3, dt=float, shape=(WIDGHT, HEIGTH))

# 每个像素所对应uv 信息
pix_uv = ti.Vector(3, dt=float, shape=(WIDGHT, HEIGTH))

face_base_normal = ti.Vector(3, dt=float, shape=12)
rotation = ti.field(dtype=float, shape=12)
# UV 贴图
uv_np = np.abs(mpig.imread("./normal.jpg"))
UV_SHAPE = uv_np.shape
uv_image = ti.Vector(3, dt=float, shape=(512, 512))
# res pic
res_normal = ti.Vector(3, dt=float, shape=(WIDGHT, HEIGTH))
uv_image.from_numpy(
    np.array(
        [[np.abs(uv_np[i, j]) / np.linalg.norm(uv_np[i, j]) for i in range(WIDGHT)] for j in range(HEIGTH)]))
# print(uv_image)
model_data.from_numpy(
    np.array([
        [0.25, 0.25, 0.75, 1],
        [0.75, 0.25, 0.75, 1],
        [0.75, 0.25, 0.25, 1],
        [0.25, 0.25, 0.25, 1],
        [0.25, 0.75, 0.75, 1],
        [0.75, 0.75, 0.75, 1],
        [0.75, 0.75, 0.25, 1],
        [0.25, 0.75, 0.25, 1],
    ])
)
tri_index.from_numpy(
    np.array([
        [1, 0, 2],
        [3, 2, 0],
        [5, 6, 4],
        [7, 4, 6],
        [4, 0, 5],
        [1, 5, 0],
        [3, 7, 2],
        [6, 2, 7],
        [2, 6, 1],
        [5, 1, 6],
        [3, 0, 7],
        [4, 7, 0],
    ])
)


# base_normal = A*B
@ti.kernel
def BaseNormal():
    for i in ti.ndrange(12):
        u = model_data[tri_index[i][1]] - model_data[tri_index[i][0]]
        v = model_data[tri_index[i][2]] - model_data[tri_index[i][1]]
        t1 = ti.Vector([u[0], u[1], u[2]])
        t2 = ti.Vector([v[0], v[1], v[2]])
        face_base_normal[i] = t1.cross(t2).normalized()


@ti.kernel
def CalNormal():
    for w, h in ti.ndrange(WIDGHT, HEIGTH):
        # 基础法向量
        base = face_base_normal[deep_index[w, h]]
        radius = ti.acos(-base[2])
        up = ti.Vector([base[1],-base[0], 0])
        q = QuaternionRotationByRadis(res_normal[w, h], up, radius)
        res_normal[w, h][0], res_normal[w, h][1], res_normal[w, h][2] = q[0], q[1], q[2]
        # print(res_normal[w, h][0], res_normal[w, h][1], res_normal[w, h][2])


@ti.kernel
def Diffuse():
    for w, h in ti.ndrange(WIDGHT, HEIGTH):
        if deep_index[w, h] != -1:
            light = ti.Vector([w / WIDGHT, h / HEIGTH, pix_uv[w, h][2]]) - LIGHT_POSITION
            pow = (res_normal[w,h].dot(light) * LIGHT_POWER) *0.3
            res_pic[w, h][0] =0.5 + pow
            res_pic[w, h][1] =0.5 + pow
            res_pic[w, h][2] =0.5 + pow
            # print(pow)
        else:
            res_pic[w, h][0] = 0
            res_pic[w, h][1] = 0
            res_pic[w, h][2] = 0

@ti.impl.pyfunc
def QuaternionToRotationMatrix(q):
    ret = ti.Vector([
        [1 - 2 * q[2] * q[2] - 2 * q[3] * q[3],
         2 * q[1] * q[2] - 2 * q[3] * q[0],
         2 * q[1] * q[3] + 2 * q[2] * q[0],
         0],
        [2 * q[1] * q[2] + 2 * q[3] * q[0],
         1 - 2 * q[1] * q[1] - 2 * q[3] * q[3],
         2 * q[2] * q[3] - 2 * q[1] * q[0],
         0],
        [2 * q[1] * q[3] - 2 * q[2] * q[0],
         2 * q[2] * q[3] + 2 * q[1] * q[0],
         1 - 2 * q[1] * q[1] - 2 * q[2] * q[2],
         0],
        [0, 0, 0, 1]

    ])
    return ret


@ti.impl.pyfunc
def RadisToQuaternion(up, r):
    w = up.normalized() * ti.sin(r / 2)
    ret = ti.Vector([ti.cos(r / 2), w[0], w[1], w[2]])
    return ret


@ti.impl.pyfunc
def QuaternionMUL(p, q):
    # print("p,q", p, q)
    ret = ti.Vector([
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[1] * q[0] + p[0] * q[1] + p[2] * q[3] - p[3] * q[2],
        p[2] * q[0] + p[0] * q[2] + p[3] * q[1] - p[1] * q[3],
        p[3] * q[0] + p[0] * q[3] + p[1] * q[2] - p[2] * q[1],
    ])
    # print("ret", ret)
    return ret


@ti.impl.pyfunc
def QuaternionRotationByRadis(point, up, r):
    p = ti.Vector([0, point[0], point[1], point[2]])
    q = RadisToQuaternion(up, r)
    # print("q",q)
    q_ = ti.Vector([q[0], -q[1], -q[2], -q[3]])
    ret = QuaternionMUL((QuaternionMUL(q, p)), q_)
    return ti.Vector([ret[1], ret[2], ret[3], 1])
    # return ret


# @ti.kernel
# def

@ti.kernel
def DeepTest():
    # O + Dt = (1 - u - v)V0 + uV1 + vV2
    # E1 = V1 - V0，E2 = V2 - V0，T = O - V0
    # P = D * E2 Q = T * E1
    for w, h in ti.ndrange(WIDGHT, HEIGTH):
        min_deep = 100.0
        index = -1
        for i in range(12):
            e1 = model_data[tri_index[i][1]] - model_data[tri_index[i][0]]
            e2 = model_data[tri_index[i][2]] - model_data[tri_index[i][0]]

            E1 = ti.Vector([e1[0], e1[1], e1[2]])
            E2 = ti.Vector([e2[0], e2[1], e2[2]])

            T = ti.Vector([w / WIDGHT - model_data[tri_index[i][0]][0], h / HEIGTH - model_data[tri_index[i][0]][1],
                           0 - model_data[tri_index[i][0]][2]])
            D = ti.Vector([0, 0, 1])
            P = D.cross(E2)
            Q = T.cross(E1)
            det = E1.dot(P)
            if det < 0:
                T = -T
            if det < 0.0001:
                continue
            n = 1 / det
            u = n * P.dot(T)
            if u > 1 or u < 0:
                continue
            v = n * Q.dot(D)
            if v > 1 or v < 0:
                continue
            if 1 - u - v < 0:
                continue
            t = n * E2.dot(Q)
            if min_deep > t:
                min_deep = t
                index = i
                pix_uv[w, h] = [u, v, t]
        deep_index[w, h] = index
        # print(index)


@ti.kernel
def Rotation():
    global model_data
    trans_matrix = ti.Vector(
        [[1, 0, 0, -0.5],
         [0, 1, 0, -0.5],
         [0, 0, 1, -0.5],
         [0, 0, 0, 1]
         ]
    )
    q = RadisToQuaternion(ti.Vector([2.,3,11,]), 0.01)
    rotation_q = QuaternionToRotationMatrix(q)
    trans = trans_matrix.inverse() @ rotation_q @ trans_matrix
    for i in range(8):
        model_data[i] = trans @ model_data[i]


@ti.kernel
def DrawPix():
    for w, h in ti.ndrange(WIDGHT, HEIGTH):
        if deep_index[w, h] == -1:
            res_normal[w, h] = [0, 0, 0]
        else:
            pix_u = int(pix_uv[w, h][0] * 512)
            pix_v = int(pix_uv[w, h][1] * 512)
            if (deep_index[w, h] % 2):
                pix_u = 512 - pix_u
                pix_v = 512 - pix_v

            res_normal[w, h] = uv_image[pix_u, pix_v]


gui2 = ti.GUI("", WIDGHT)
slider = gui.slider("light-power", 0, 10, step=1)

while 1:

    DeepTest()
    BaseNormal()
    DrawPix()
    CalNormal()
    Diffuse()
    gui2.set_image(res_normal)
    gui.set_image(res_pic)
    Rotation()
    gui.show()
    gui2.show()
