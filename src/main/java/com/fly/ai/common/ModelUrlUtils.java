package com.fly.ai.common;

import lombok.SneakyThrows;
import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ClassPathResource;

import java.net.URI;

/**
 * @author tiny tiny
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/19 12:16
 */
@UtilityClass
@Slf4j
public class ModelUrlUtils {


    /**
     * 获取模型url，如果是http或file开头，直接返回
     *
     * @param name 模型名称
     * @return url
     */
    @SneakyThrows
    public static String getRealUrl(String name) {
        if (name.startsWith("http") || name.startsWith("file:")) {
            log.debug("model url is {}", name);
            return name;
        }

        URI uri = new ClassPathResource(name).getURI();
        log.debug("model uri of {} is {}", name, uri);
        if (uri.toString().startsWith("jar:")) {
            return "jar://" + name;
        }

        return uri.toString();
    }

}
