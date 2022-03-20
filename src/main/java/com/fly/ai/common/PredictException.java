package com.fly.ai.common;

/**
 * 推理过程出错
 *
 * @author TinyThing
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/20 13:18
 */
public class PredictException extends RuntimeException {

    public PredictException() {
    }

    public PredictException(Throwable exception) {
        super(exception);
    }

    public PredictException(String message, Throwable cause) {
        super(message, cause);
    }
}
