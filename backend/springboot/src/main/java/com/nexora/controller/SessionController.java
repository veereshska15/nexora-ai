package com.nexora.controller;

import com.nexora.dto.response.ApiResponse;
import com.nexora.dto.response.SessionResponse;
import com.nexora.entity.UserSession;
import com.nexora.security.UserPrincipal;
import com.nexora.service.SessionService;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1/sessions")
@RequiredArgsConstructor
public class SessionController {

    private final SessionService sessionService;

    @GetMapping("/me")
    public ResponseEntity<ApiResponse<List<SessionResponse>>> getMySessions(
            @AuthenticationPrincipal UserPrincipal currentUser,
            HttpServletRequest servletRequest) {

        String requestId = servletRequest.getHeader("X-Request-ID");
        List<UserSession> sessions = sessionService.getUserSessions(currentUser.getId());
        List<SessionResponse> dtos = sessions.stream().map(s -> SessionResponse.builder()
                .id(s.getId())
                .userId(s.getUser().getId())
                .ipAddress(s.getIpAddress())
                .userAgent(s.getUserAgent())
                .isRevoked(s.getIsRevoked())
                .expiresAt(s.getExpiresAt())
                .createdAt(s.getCreatedAt())
                .build()
        ).collect(Collectors.toList());

        return ResponseEntity.ok(ApiResponse.success(dtos, "Active sessions fetched", requestId != null ? requestId : "sess"));
    }

    @PostMapping("/revoke/{id}")
    public ResponseEntity<ApiResponse<String>> revokeSession(
            @PathVariable UUID id,
            HttpServletRequest servletRequest) {

        String requestId = servletRequest.getHeader("X-Request-ID");
        sessionService.revokeSession(id);
        return ResponseEntity.ok(ApiResponse.success("Session revoked", "Revoked session successfully", requestId != null ? requestId : "revoke"));
    }
}
