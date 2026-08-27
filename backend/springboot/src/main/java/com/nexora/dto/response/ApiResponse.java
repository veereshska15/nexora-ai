package com.nexora.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ApiResponse<T> {

    private Boolean success;
    private String message;
    private T data;
    private Object error;
    private String requestId;
    @Builder.Default
    private Instant timestamp = Instant.now();

    public static <T> ApiResponse<T> success(T data, String message, String requestId) {
        return ApiResponse.<T>builder()
                .success(true)
                .message(message)
                .data(data)
                .requestId(requestId)
                .timestamp(Instant.now())
                .build();
    }

    public static <T> ApiResponse<T> error(Object error, String message, String requestId) {
        return ApiResponse.<T>builder()
                .success(false)
                .message(message)
                .error(error)
                .requestId(requestId)
                .timestamp(Instant.now())
                .build();
    }
}
