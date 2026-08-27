package com.nexora.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SessionResponse {

    private UUID id;
    private UUID userId;
    private String ipAddress;
    private String userAgent;
    private Boolean isRevoked;
    private Instant expiresAt;
    private Instant createdAt;
}
