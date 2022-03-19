package com.fly.ai.yolo;

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
 * @since 2022/3/19 13:38
 */
@RestController
@RequestMapping("yolo")
@RequiredArgsConstructor
@Slf4j
public class YoloController {

    private final YoloUtils yoloUtils;

    @PostMapping("image")
    @ApiOperation("图片对象检测，返回结果图片")
    public void ocr(@RequestPart MultipartFile file, HttpServletResponse response) throws IOException {

        BufferedImage image = ImageIO.read(file.getInputStream());
        BufferedImage result = yoloUtils.getResultImage(image);

        response.setContentType("image/png");
        ServletOutputStream os = response.getOutputStream();
        ImageIO.write(result, "PNG", os);
        os.flush();
    }


    @PostMapping
    @ApiOperation("图片对象检测，返回结果对象")
    public List<DetectObjectDto> ocr(@RequestPart MultipartFile file) throws IOException {
        BufferedImage image = ImageIO.read(file.getInputStream());
        DetectedObjects result = yoloUtils.detect(image);

        return result.items().stream().map(DetectObjectDto::new).collect(Collectors.toList());
    }



}
