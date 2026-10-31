SELECT
    customer_id,
    SUM(order_amount) AS total_spent,
    COUNT(*) AS order_count
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spent DESC;

-- Query 2: Revenue by city and product category

SELECT
    c.city,
    p.category,
    SUM(o.order_amount) AS total_revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Query 3: Products priced above average catalogue price

SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;

-- Query 4: Customers with no completed orders

SELECT
    c.customer_id,
    c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Query 5: Customers spending above average

WITH customer_spending AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spent
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spending AS (
    SELECT AVG(total_spent) AS avg_spent
    FROM customer_spending
)
SELECT
    cs.customer_id,
    cs.total_spent,
    av.avg_spent
FROM customer_spending cs
CROSS JOIN average_spending av
WHERE cs.total_spent > av.avg_spent
ORDER BY cs.total_spent DESC;

-- Query 6: Customer ranking within each city

WITH customer_spending AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spent
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spent,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spent DESC
    ) AS spending_rank,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spent DESC
    ) AS dense_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spent DESC
    ) AS row_number
FROM customer_spending
ORDER BY city, spending_rank, customer_id;

-- Query 7: Daily revenue with running total

WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS order_day,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    order_day,
    daily_revenue,
    SUM(daily_revenue) OVER (
        ORDER BY order_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM daily_revenue
ORDER BY order_day;