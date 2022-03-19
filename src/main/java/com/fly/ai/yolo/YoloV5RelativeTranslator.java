package com.fly.ai.yolo;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.List;

/**
 * DJL自带的translator返回的对象外框是绝对坐标，我们转为相对坐标方便统一处理
 * 采用静态代理
 *
 * @author TinyThing
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/19 21:14
 */
public class YoloV5RelativeTranslator implements Translator<Image, DetectedObjects> {

    private final YoloProperties yoloProperties;

    private final Translator<Image, DetectedObjects> delegated;

    public YoloV5RelativeTranslator(Translator<Image, DetectedObjects> translator, YoloProperties yolo) {
        this.delegated = translator;
        this.yoloProperties = yolo;
    }

    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws Exception {
        DetectedObjects output = delegated.processOutput(ctx, list);
        List<String> classList = new ArrayList<>();
        List<Double> probList = new ArrayList<>();
        List<BoundingBox> rectList = new ArrayList<>();

        final Integer width = yoloProperties.getWidth();
        final Integer height = yoloProperties.getHeight();

        final List<DetectedObjects.DetectedObject> items = output.items();
        items.forEach(item -> {
            classList.add(item.getClassName());
            probList.add(item.getProbability());

            Rectangle b = item.getBoundingBox().getBounds();
            Rectangle newBox = new Rectangle(b.getX() / width, b.getY() / height, b.getWidth() / width, b.getHeight() / height);

            rectList.add(newBox);
        });
        return new DetectedObjects(classList, probList, rectList);
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
        return delegated.processInput(ctx, input);
    }

    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        delegated.prepare(ctx);
    }

    @Override
    public Batchifier getBatchifier() {
        return delegated.getBatchifier();
    }
}
