package com.nexora.security;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nexora.dto.response.ApiResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.MediaType;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.AuthenticationEntryPoint;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
public class JwtAuthenticationEntryPoint implements AuthenticationEntryPoint {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public void commence(HttpServletRequest request,
                         HttpServletResponse response,
                         AuthenticationException authException) throws IOException {

        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);

        String requestId = request.getHeader("X-Request-ID");
        ApiResponse<Object> apiResponse = ApiResponse.error(
                "UNAUTHORIZED",
                "Full authentication is required to access this resource: " + authException.getMessage(),
                requestId != null ? requestId : "auth-err"
        );

        objectMapper.writeValue(response.getOutputStream(), apiResponse);
    }
}
