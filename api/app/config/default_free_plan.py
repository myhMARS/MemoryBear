"""
社区版默认免费套餐配置
当无法从 SaaS 版获取 premium 模块时，使用此配置作为兜底
"""

DEFAULT_FREE_PLAN = {
    "name": "记忆体验版",
    "name_en": "Memory Experience",
    "category": "saas_personal",
    "tier_level": 0,
    "version": "1.0",
    "status": True,
    "price": 0,
    "billing_cycle": "permanent_free",
    "core_value": "感受永久记忆",
    "core_value_en": "Experience Permanent Memory",
    "tech_support": "社群交流",
    "tech_support_en": "Community Support",
    "sla_compliance": "无",
    "sla_compliance_en": "None",
    "page_customization": "无",
    "page_customization_en": "None",
    "theme_color": "#64748B",
    "quotas": {
        "workspace_quota": 10,
        "skill_quota": 50,
        "app_quota": 20,
        "knowledge_capacity_quota": 30,
        "memory_engine_quota": 10,
        "end_user_quota": 50,
        "ontology_project_quota": 30,
        "model_quota": 10,
        "api_ops_rate_limit": 50,
    },
}
