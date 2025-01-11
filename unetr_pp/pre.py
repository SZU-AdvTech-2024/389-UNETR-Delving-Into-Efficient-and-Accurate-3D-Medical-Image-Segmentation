import SimpleITK as sitk
import os

def n4_bias_field_correction(input_image):
    """
    使用N4ITK算法对MRI图像进行偏置场校正
    :param input_image: SimpleITK图像对象
    :return: 校正后的图像
    """
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(input_image)
    return corrected_image

def affine_registration(fixed_image, moving_image):
    """
    使用仿射变换将移动图像配准到固定图像
    :param fixed_image: 固定图像（SimpleITK图像对象）
    :param moving_image: 移动图像（SimpleITK图像对象）
    :return: 配准后的图像和变换参数
    """
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.AffineTransform(fixed_image.GetDimension())
    )
    registration_method.SetInitialTransform(initial_transform)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    registered_image = sitk.Resample(
        moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID()
    )
    return registered_image, final_transform

def b_spline_registration(fixed_image, moving_image):
    """
    使用B样条自由形变（FFD）配准将移动图像配准到固定图像
    :param fixed_image: 固定图像（SimpleITK图像对象）
    :param moving_image: 移动图像（SimpleITK图像对象）
    :return: 配准后的图像和变换参数
    """
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    transform_domain_mesh_size = [8] * fixed_image.GetDimension()
    initial_transform = sitk.BSplineTransformInitializer(
        fixed_image, transform_domain_mesh_size
    )
    registration_method.SetInitialTransform(initial_transform)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    registered_image = sitk.Resample(
        moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID()
    )
    return registered_image, final_transform

def preprocess_mri_image(input_image_path, fixed_image_path, output_dir):
    """
    对MRI图像进行完整的预处理，包括N4偏置场校正、仿射变换空间标准化和B样条自由形变配准
    :param input_image_path: 输入MRI图像路径
    :param fixed_image_path: 标准空间图像路径
    :param output_dir: 输出目录
    :return: 预处理后的图像路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取输入图像和标准空间图像
    input_image = sitk.ReadImage(input_image_path)
    fixed_image = sitk.ReadImage(fixed_image_path)
    
    # 1. N4偏置场校正
    corrected_image = n4_bias_field_correction(input_image)
    corrected_image_path = os.path.join(output_dir, "corrected_image.nii.gz")
    sitk.WriteImage(corrected_image, corrected_image_path)
    
    # 2. 仿射变换空间标准化
    affine_registered_image, _ = affine_registration(fixed_image, corrected_image)
    affine_registered_image_path = os.path.join(output_dir, "affine_registered_image.nii.gz")
    sitk.WriteImage(affine_registered_image, affine_registered_image_path)
    
    # 3. B样条自由形变配准
    b_spline_registered_image, _ = b_spline_registration(fixed_image, affine_registered_image)
    b_spline_registered_image_path = os.path.join(output_dir, "b_spline_registered_image.nii.gz")
    sitk.WriteImage(b_spline_registered_image, b_spline_registered_image_path)
    
    # 返回最终处理结果路径
    return b_spline_registered_image_path

# 示例调用
if __name__ == "__main__":
    # 输入路径
    input_image_path = "input_image.nii.gz"  # 输入MRI图像
    fixed_image_path = "fixed_image.nii.gz"  # 标准空间图像
    output_dir = "output"  # 输出目录
    
    # 调用预处理函数
    final_image_path = preprocess_mri_image(input_image_path, fixed_image_path, output_dir)
    print(f"预处理完成，最终图像保存至: {final_image_path}")