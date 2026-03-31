"""
test_tools.py - 測試所有 Agent 工具函數
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTourGuideTools:
    """導覽員工具測試"""

    def test_get_factory_info_valid(self):
        from factory_tour_agent import get_factory_info

        result = get_factory_info.invoke({"area_name": "大廳"})
        assert "大廳" in result
        assert "訪客" in result

    def test_get_factory_info_invalid(self):
        from factory_tour_agent import get_factory_info

        result = get_factory_info.invoke({"area_name": "不存在的區域"})
        assert "找不到" in result
        assert "可用區域" in result

    def test_get_all_areas(self):
        from factory_tour_agent import get_all_areas

        result = get_all_areas.invoke({})
        assert "大廳" in result
        assert "組裝線A" in result
        assert "品管室" in result
        assert "倉儲區" in result
        assert "會議室" in result

    def test_get_route_info_all(self):
        from factory_tour_agent import get_route_info

        result = get_route_info.invoke({})
        assert "標準導覽路線" in result
        assert "快速導覽路線" in result

    def test_get_route_info_specific(self):
        from factory_tour_agent import get_route_info

        result = get_route_info.invoke({"route_name": "標準導覽路線"})
        assert "45 分鐘" in result
        assert "大廳" in result


class TestSafetyTools:
    """安全專家工具測試"""

    def test_get_safety_rules_valid(self):
        from factory_tour_agent import get_safety_rules

        result = get_safety_rules.invoke({"area_name": "組裝線A"})
        assert "防靜電" in result
        assert "護目鏡" in result

    def test_get_safety_rules_no_rules(self):
        from factory_tour_agent import get_safety_rules

        result = get_safety_rules.invoke({"area_name": "會議室"})
        assert "沒有特殊安全規範" in result

    def test_get_safety_rules_invalid(self):
        from factory_tour_agent import get_safety_rules

        result = get_safety_rules.invoke({"area_name": "XXXX"})
        assert "找不到" in result

    def test_get_all_safety_rules(self):
        from factory_tour_agent import get_all_safety_rules

        result = get_all_safety_rules.invoke({})
        assert "組裝線A" in result
        assert "品管室" in result
        assert "⚠️" in result

    def test_get_emergency_info(self):
        from factory_tour_agent import get_emergency_info

        result = get_emergency_info.invoke({})
        assert "緊急" in result
        assert "AED" in result
        assert "集合" in result


class TestTechExpertTools:
    """技術專家工具測試"""

    def test_get_equipment_details(self):
        from factory_tour_agent import get_equipment_details

        result = get_equipment_details.invoke({"area_name": "組裝線A"})
        assert "SMT" in result
        assert "技術規格" in result or "設備" in result

    def test_get_equipment_details_no_specs(self):
        from factory_tour_agent import get_equipment_details

        result = get_equipment_details.invoke({"area_name": "大廳"})
        assert "大廳" in result

    def test_get_production_metrics(self):
        from factory_tour_agent import get_production_metrics

        result = get_production_metrics.invoke({})
        assert "組裝線A" in result or "品管室" in result
        assert "指標" in result or "規格" in result

    def test_compare_areas(self):
        from factory_tour_agent import compare_areas

        result = compare_areas.invoke({"area1": "組裝線A", "area2": "品管室"})
        assert "組裝線A" in result
        assert "品管室" in result
        assert "比較" in result or "vs" in result

    def test_compare_areas_invalid(self):
        from factory_tour_agent import compare_areas

        result = compare_areas.invoke({"area1": "XXX", "area2": "YYY"})
        assert "找不到" in result


class TestQATools:
    """QA Agent 工具測試"""

    def test_search_faq_by_keyword(self):
        from factory_tour_agent import search_faq

        result = search_faq.invoke({"keyword": "停車"})
        assert "停車" in result

    def test_search_faq_by_keyword_en(self):
        from factory_tour_agent import search_faq

        result = search_faq.invoke({"keyword": "parking"})
        assert "停車" in result

    def test_search_faq_not_found(self):
        from factory_tour_agent import search_faq

        result = search_faq.invoke({"keyword": "asdfghjkl"})
        assert "找不到" in result

    def test_get_all_faq(self):
        from factory_tour_agent import get_all_faq

        result = get_all_faq.invoke({})
        assert "常見問題" in result
        # 應有多個問題
        assert "參觀" in result or "時間" in result

    def test_get_visitor_guidelines(self):
        from factory_tour_agent import get_visitor_guidelines

        result = get_visitor_guidelines.invoke({})
        assert "訪客" in result or "須知" in result
