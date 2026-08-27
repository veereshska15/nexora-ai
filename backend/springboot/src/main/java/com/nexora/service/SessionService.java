package com.nexora.service;

import com.nexora.entity.User;
import com.nexora.entity.UserSession;
import com.nexora.exception.ResourceNotFoundException;
import com.nexora.repository.SessionRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class SessionService {

    private final SessionRepository sessionRepository;
    private final PasswordEncoder passwordEncoder;

    @Transactional
    public UserSession createSession(User user, String refreshToken, String ipAddress, String userAgent) {
        String tokenHash = passwordEncoder.encode(refreshToken);
        Instant expiresAt = Instant.now().plus(7, ChronoUnit.DAYS);

        UserSession session = UserSession.builder()
                .user(user)
                .refreshTokenHash(tokenHash)
                .ipAddress(ipAddress)
                .userAgent(userAgent)
                .isRevoked(false)
                .expiresAt(expiresAt)
                .build();

        return sessionRepository.save(session);
    }

    @Transactional(readOnly = true)
    public List<UserSession> getUserSessions(UUID userId) {
        return sessionRepository.findByUserIdAndIsRevokedFalse(userId);
    }

    @Transactional
    public void revokeSession(UUID sessionId) {
        UserSession session = sessionRepository.findById(sessionId)
                .orElseThrow(() -> new ResourceNotFoundException("Session not found: " + sessionId));

        session.setIsRevoked(true);
        sessionRepository.save(session);
    }
}
