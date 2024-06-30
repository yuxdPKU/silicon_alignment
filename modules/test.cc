#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <iostream>

int test() {
    // 初始化 TGeoManager
    TGeoManager *geom = new TGeoManager("geom", "Geometry manager");

    // 定义从坐标系 A 到坐标系 B 的变换
    TGeoTranslation translation(10, 20, 30);
    TGeoRotation rotation;
    rotation.SetAngles(45, 45, 45); // 设置旋转角度为45度
    TGeoCombiTrans transAB(translation, rotation);

    // 将 TGeoCombiTrans 转换为 TGeoHMatrix
    TGeoHMatrix hMatrixAB = transAB;

    // 打印从 A 到 B 的变换矩阵
    std::cout << "Transformation from A to B:" << std::endl;
    hMatrixAB.Print();

    // 获取从 B 到 A 的逆变换
    TGeoHMatrix hMatrixBA;
    hMatrixBA.SetRotation(hMatrixAB.GetRotationMatrix());
    hMatrixBA.SetTranslation(hMatrixAB.GetTranslation());
    //hMatrixBA.Invert();

    // 打印从 B 到 A 的逆变换矩阵
    std::cout << "Inverse transformation from B to A:" << std::endl;
    hMatrixBA.Print();

    // 应用变换和逆变换到一个点
    Double_t pointA[3] = {1, 1, 1};
    Double_t pointB[3];
    Double_t pointA_transformed[3];

    // 从 A 到 B 变换
    hMatrixAB.LocalToMaster(pointA, pointB);
    std::cout << "Point in A: (" << pointA[0] << ", " << pointA[1] << ", " << pointA[2] << ")" << std::endl;
    std::cout << "Transformed point in B: (" << pointB[0] << ", " << pointB[1] << ", " << pointB[2] << ")" << std::endl;

    // 从 B 到 A 逆变换
    hMatrixBA.MasterToLocal(pointB, pointA_transformed);
    std::cout << "Transformed back point in A: (" << pointA_transformed[0] << ", " << pointA_transformed[1] << ", " << pointA_transformed[2] << ")" << std::endl;

    // 检查精度
    std::cout << "Precision check:" << std::endl;
    std::cout << "Delta X: " << pointA[0] - pointA_transformed[0] << std::endl;
    std::cout << "Delta Y: " << pointA[1] - pointA_transformed[1] << std::endl;
    std::cout << "Delta Z: " << pointA[2] - pointA_transformed[2] << std::endl;

    return 0;
}

