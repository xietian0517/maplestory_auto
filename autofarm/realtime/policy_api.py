"""Safe API diagnostics: retain known codes, never transport messages or keys."""
import json
import socket
import ssl
import urllib.error
import urllib.request

from .semantic import PlannerError


MESSAGES = {
    'missing_key': ('尚未填写 API Key。', '点击 AI 配置，填写 Key。'),
    'invalid_api_key': ('API Key 无效、已过期或已撤销。', '在开发者平台确认或重新创建 Key。'),
    'insufficient_quota': ('API 余额或可用额度不足。', '在所选服务商的 API 控制台检查余额和额度。'),
    'unsupported_vision_model': ('所选 DeepSeek 模型未配置图像支持。', '将行动模型设为 deepseek-flash。'),
    'organization_spend_limit_exceeded': ('已达到组织的费用上限。', '检查开发者平台的组织费用上限。'),
    'project_spend_limit_exceeded': ('已达到项目的费用上限。', '检查开发者平台的项目费用上限。'),
    'organization_usage_limit_exceeded': ('已达到组织的 API 用量上限。', '检查开发者平台的用量限制。'),
    'model_not_found': ('模型名称不存在，或当前项目无权使用。', '在 AI 配置中确认账号可用的模型名称。'),
    'permission_denied': ('请求被拒绝，当前 Key 或项目权限不足。', '检查 Key 的权限、项目及账号访问条件。'),
    'unsupported_country_region_territory': ('API 服务不支持当前请求所在地区。', '检查 OpenAI 官方支持的国家和地区。'),
    'invalid_request': ('模型请求参数被拒绝。', '查看日志中的 HTTP 状态和参数分类。'),
    'rate_limit_exceeded': ('API 请求受到速率限制。', '稍后重试或检查项目的请求速率限制。'),
    'server_error': ('模型 API 服务暂时不可用。', '稍后重试。'),
    'timeout': ('模型请求超时。', '检查网络；较慢模型可能需要调整超时和有效期限。'),
    'tls_error': ('HTTPS 证书验证或连接失败。', '检查系统时间、证书和网络设置。'),
    'connection_error': ('无法连接到模型 API。', '检查本机网络、代理和防火墙设置。'),
    'invalid_json': ('API 未返回有效 JSON。', '检查网络中间服务及 API 返回格式。'),
    'incomplete_output': ('模型未完成动作回复。', '查看未完成原因；输出预算可能不足。'),
    'output_token_limit': ('推理及回复耗尽输出预算。', '使用低推理强度或提高 max_output_tokens。'),
    'content_filter': ('本次回复未完成，服务返回内容过滤标记。', '查看请求场景，暂不执行该回复。'),
    'refusal_or_missing_output': ('模型没有返回单一可执行动作。', '暂不执行；检查模型是否支持结构化动作输出。'),
    'invalid_action': ('模型动作不符合协议或当前请求标识。', '暂不执行；查看请求与动作校验记录。'),
    'unknown_error': ('检测发生未分类错误。', '查看本轮错误类型，不要把密钥发送到聊天。'),
}

PERMANENT = {'missing_key', 'invalid_api_key', 'insufficient_quota',
    'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
    'organization_usage_limit_exceeded', 'model_not_found', 'permission_denied',
    'unsupported_country_region_territory', 'invalid_request','unsupported_vision_model'}
SAFE_PARAMS = {'model', 'input', 'max_output_tokens', 'reasoning', 'reasoning.effort',
    'text.format', 'text.format.schema', 'store', 'service_tier'}


class PolicyAPIError(PlannerError):
    def __init__(self, code, http_status=None, param=None):
        self.code = code if code in MESSAGES else 'unknown_error'
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        self.param = param if isinstance(param,str) and param in SAFE_PARAMS else None
        super().__init__(self.code)

    def data(self):
        description, remedy = MESSAGES[self.code]
        return dict(code=self.code, http_status=self.http_status, param=self.param,
                    description=description, remedy=remedy, retryable=self.code not in PERMANENT)


def http_error(error):
    code = {400:'invalid_request',401:'invalid_api_key',402:'insufficient_quota',403:'permission_denied',
            404:'model_not_found',429:'rate_limit_exceeded'}.get(error.code, 'server_error')
    param = None
    try:
        body = json.loads(error.read(65536))
        detail = body.get('error')
        if isinstance(detail,dict):
            for field in ('code','type'):
                value=detail.get(field)
                if isinstance(value,str) and value in MESSAGES:
                    code=value
                    break
            param=detail.get('param')
    except Exception:
        pass
    return PolicyAPIError(code,error.code,param)


def request_json(request, timeout, opener=None):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    opener=opener or urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request,timeout=timeout) as response:
            data=json.load(response)
    except urllib.error.HTTPError as error:
        raise http_error(error) from None
    except (TimeoutError,socket.timeout):
        raise PolicyAPIError('timeout') from None
    except urllib.error.URLError as error:
        reason=error.reason
        code='timeout' if isinstance(reason,TimeoutError) else 'tls_error' if isinstance(reason,ssl.SSLError) else 'connection_error'
        raise PolicyAPIError(code) from None
    except ssl.SSLError:
        raise PolicyAPIError('tls_error') from None
    except OSError:
        raise PolicyAPIError('connection_error') from None
    except (ValueError,TypeError):
        raise PolicyAPIError('invalid_json') from None
    if not isinstance(data,dict):raise PolicyAPIError('invalid_json')
    return data
