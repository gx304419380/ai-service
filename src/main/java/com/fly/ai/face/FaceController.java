package com.fly.ai.face;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import com.fly.ai.common.DetectObjectDto;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletResponse;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author TinyThing
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/20 13:23
 */
@RestController
@RequestMapping("face")
@RequiredArgsConstructor
@Slf4j
public class FaceController {

    private final FaceDetectService faceDetectService;

    private final FaceCompareService faceCompareService;

    @PostMapping("detect/image")
    @ApiOperation("检测图片中的人脸，返回结果图片")
    public void face(@RequestPart MultipartFile file, HttpServletResponse response) throws IOException {

        BufferedImage image = ImageIO.read(file.getInputStream());

        BufferedImage resultImage = faceDetectService.detectAndDraw(image);

        response.setContentType("image/png");
        ServletOutputStream os = response.getOutputStream();
        ImageIO.write(resultImage, "PNG", os);
        os.flush();
    }


    @PostMapping("detect")
    @ApiOperation("检测图片人脸，返回结果对象")
    public List<DetectObjectDto> face(@RequestPart MultipartFile file) throws IOException {
        BufferedImage image = ImageIO.read(file.getInputStream());
        Image img = ImageFactory.getInstance().fromImage(image);
        DetectedObjects result = faceDetectService.detect(img);

        return result.items().stream().map(DetectObjectDto::new).collect(Collectors.toList());
    }


    @PostMapping("feature")
    @ApiOperation("检测图片人脸，返回各个人脸特征值")
    public List<DetectObjectDto> faceFeature(@RequestPart MultipartFile file) throws IOException {
        BufferedImage image = ImageIO.read(file.getInputStream());
        Image img = ImageFactory.getInstance().fromImage(image);

        return faceCompareService.getFeatureList(img);
    }



}
