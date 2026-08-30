CREATE TABLE IF NOT EXISTS customers (
    customer_id       SERIAL PRIMARY KEY,
    name              VARCHAR(120) NOT NULL,
    email             VARCHAR(120) UNIQUE NOT NULL,
    phone             VARCHAR(15) NOT NULL,
    plan_type         VARCHAR(20) NOT NULL,
    monthly_charge    NUMERIC(10, 2) NOT NULL,
    tenure_months     INTEGER DEFAULT 0,
    total_charges     NUMERIC(10, 2) DEFAULT 0,
    contract_type     VARCHAR(20) NOT NULL,
    payment_method    VARCHAR(20) NOT NULL,
    num_tickets       INTEGER DEFAULT 0,
    avg_satisfaction  NUMERIC(3, 2) DEFAULT 3.0,
    is_active         BOOLEAN DEFAULT TRUE,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id       SERIAL PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    category        VARCHAR(10) NOT NULL,
    description     TEXT NOT NULL,
    priority        VARCHAR(10) DEFAULT 'medium',
    status          VARCHAR(20) DEFAULT 'open',
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    resolved_at     TIMESTAMP,
    satisfaction    NUMERIC(3, 2)
);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id   SERIAL PRIMARY KEY,
    ticket_id        INTEGER NOT NULL REFERENCES tickets(ticket_id),
    agent_response   TEXT,
    customer_msg     TEXT NOT NULL,
    sentiment        VARCHAR(15),
    timestamp        TIMESTAMP DEFAULT NOW(),
    resolution_time  NUMERIC(6, 2)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   SERIAL PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    churn_prob      NUMERIC(5, 4) NOT NULL,
    risk_level      VARCHAR(10) NOT NULL,
    model_version   VARCHAR(20) DEFAULT 'v1',
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id     VARCHAR(50) PRIMARY KEY,
    customer_id    INTEGER REFERENCES customers(customer_id),
    conversation   TEXT DEFAULT '[]',
    tokens_used    INTEGER DEFAULT 0,
    is_active      BOOLEAN DEFAULT TRUE,
    started_at     TIMESTAMP DEFAULT NOW(),
    ended_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id                SERIAL PRIMARY KEY,
    username          VARCHAR(50) UNIQUE NOT NULL,
    hashed_password   VARCHAR(200) NOT NULL,
    role              VARCHAR(20) DEFAULT 'customer',
    is_active         BOOLEAN DEFAULT TRUE
);

-- =============================================================================
-- STORED PROCEDURE
-- =============================================================================

CREATE OR REPLACE PROCEDURE sp_close_ticket(
    p_ticket_id INTEGER,
    p_satisfaction NUMERIC(3, 2)
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_customer_id INTEGER;
BEGIN
    -- 1. Cierra el ticket
    UPDATE tickets
    SET status = 'resolved',
        resolved_at = NOW(),
        satisfaction = p_satisfaction
    WHERE ticket_id = p_ticket_id
    RETURNING customer_id INTO v_customer_id;

    IF v_customer_id IS NULL THEN
        RAISE EXCEPTION 'Ticket % no encontrado', p_ticket_id;
    END IF;

    -- 2. Recalcula el promedio de satisfacción del cliente sobre TODOS

    UPDATE customers
    SET avg_satisfaction = (
        SELECT AVG(satisfaction)
        FROM tickets
        WHERE customer_id = v_customer_id AND satisfaction IS NOT NULL
    )
    WHERE customer_id = v_customer_id;
END;
$$;


-- =============================================================================
-- FUNCIÓN 
-- =============================================================================

CREATE OR REPLACE FUNCTION fn_customers_at_risk(p_min_tickets INTEGER DEFAULT 2)
RETURNS TABLE (
    customer_id INTEGER,
    name VARCHAR,
    num_tickets INTEGER,
    avg_satisfaction NUMERIC,
    contract_type VARCHAR
)
LANGUAGE sql
AS $$
    SELECT customer_id, name, num_tickets, avg_satisfaction, contract_type
    FROM customers
    WHERE is_active = TRUE
      AND num_tickets >= p_min_tickets
      AND avg_satisfaction < 3.0
    ORDER BY avg_satisfaction ASC;
$$;
