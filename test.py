import taichi as ti

ti.init(arch=ti.gpu)



@ti.impl.pyfunc
def Test1(a):
    print(a.normalized())

@ti.impl.pyfunc
def Test(a):
    Test1(a)

@ti.kernel
def TestK():
    print(Test(ti.Vector([0, 0, 1])))
    print(Test(ti.Vector([0., 0., 1.])))
TestK()
